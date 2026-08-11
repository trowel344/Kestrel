import json
from types import SimpleNamespace

import pytest

from kestrel import model_store
from kestrel.model_store import (
    ModelStoreError,
    _ollama_list,
    _ollama_manifest_paths,
    _resolve_blob_from_manifest,
    choose_default_gguf,
    discover_local_models,
    hf_snapshot_dir,
)


def test_ollama_list_parses_rows(monkeypatch):
    output = (
        "NAME             ID              SIZE      MODIFIED       \n"
        "qwen3.6:35B      07d35212591f    23 GB     39 minutes ago\n"
        "gemma4:31b       6316f0629137    19 GB     34 hours ago\n"
    )
    monkeypatch.setattr(
        model_store,
        "_run",
        lambda *a, **k: SimpleNamespace(stdout=output, returncode=0),
    )
    rows = _ollama_list()
    assert [r.name for r in rows] == ["qwen3.6:35B", "gemma4:31b"]
    assert rows[0].model_id == "07d35212591f"
    assert rows[0].size == "23 GB"


def test_ollama_list_ignores_header_and_blank(monkeypatch):
    monkeypatch.setattr(
        model_store,
        "_run",
        lambda *a, **k: SimpleNamespace(stdout="NAME   ID   SIZE   MODIFIED\n\n", returncode=0),
    )
    assert _ollama_list() == []


def test_huggingface_search_skips_non_object_rows(monkeypatch):
    monkeypatch.setattr(model_store, "_hf_cli", lambda modern_only=False: "/usr/bin/hf")
    monkeypatch.setattr(
        model_store,
        "_run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(["bad", {"id": "owner/model", "downloads": 2, "likes": 1, "tags": [None, "license:mit"]}])
        ),
    )

    assert model_store.search_huggingface("model") == [
        {
            "id": "owner/model",
            "downloads": 2,
            "likes": 1,
            "last_modified": None,
            "license": "mit",
        }
    ]


def test_huggingface_file_listing_rejects_invalid_size(monkeypatch):
    monkeypatch.setattr(model_store, "_hf_cli", lambda modern_only=False: "/usr/bin/hf")
    monkeypatch.setattr(
        model_store,
        "_run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps([{"path": "model.gguf", "size": "huge"}])),
    )

    with pytest.raises(ModelStoreError, match="invalid size"):
        model_store.list_huggingface_ggufs("owner/model")


def test_missing_command_raises_model_error():
    with pytest.raises(ModelStoreError, match="required command is not installed"):
        model_store._run(["definitely-not-a-real-command-xyz"])


def test_command_spawn_oserror_is_typed(monkeypatch):
    monkeypatch.setattr(
        model_store.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("not executable")),
    )

    with pytest.raises(ModelStoreError, match="could not run hf"):
        model_store._run(["hf", "models", "ls"])


def test_hf_repo_validation():
    with pytest.raises(ModelStoreError, match="OWNER/REPOSITORY"):
        model_store.pull_huggingface("single-name", dry_run=True)


def test_hf_conflicting_file_include():
    with pytest.raises(ModelStoreError, match="either"):
        model_store.pull_huggingface("owner/repo", filename="x.gguf", include="*.gguf", dry_run=True)


def test_hf_filename_option_guard():
    with pytest.raises(ModelStoreError, match="may not start with"):
        model_store.pull_huggingface("owner/repo", filename="-o", dry_run=True)
    with pytest.raises(ModelStoreError, match="within the Hugging Face repository"):
        model_store.pull_huggingface("owner/repo", filename="../secret.gguf", dry_run=True)


def test_hf_snapshot_ref_cannot_escape_snapshot_root(tmp_path):
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("{}")
    (cache / "refs").mkdir(parents=True)
    (cache / "snapshots").mkdir()
    (cache / "refs" / "main").write_text("../../outside")

    assert hf_snapshot_dir(cache) is None


def test_hf_snapshot_directory_symlink_cannot_escape_root(tmp_path):
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("{}")
    snapshots = cache / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "escaped").symlink_to(outside, target_is_directory=True)

    assert hf_snapshot_dir(cache) is None


def test_ollama_manifest_uses_native_bare_tag_and_validated_digest(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    digest = "sha256:" + "a" * 64
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / "demo" / "latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": digest}]}))
    blob = root / "blobs" / digest.replace(":", "-")
    blob.parent.mkdir()
    blob.write_bytes(b"GGUF")

    assert _resolve_blob_from_manifest("demo") == blob.resolve()


def test_ollama_manifest_ignores_malformed_layer_schema(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / "demo" / "latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": ["not-an-object"]}))

    assert _resolve_blob_from_manifest("demo") is None


def test_ollama_manifest_supports_qualified_reference(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    root.mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(root))

    paths = _ollama_manifest_paths("registry.ollama.ai/acme/demo:v2")

    assert paths[0][1] == root / "manifests" / "registry.ollama.ai" / "acme" / "demo" / "v2"


@pytest.mark.parametrize("reference", ["../secret", "demo:../secret", "/absolute", "-option"])
def test_ollama_manifest_rejects_path_traversal(reference, tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    root.mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(root))

    with pytest.raises(ModelStoreError, match="invalid Ollama"):
        _ollama_manifest_paths(reference)


def test_choose_default_gguf_prefers_unsplit_only(tmp_path):
    single = tmp_path / "model.gguf"
    single.write_bytes(b"GGUF" + b"\x00" * 8)
    chosen = choose_default_gguf([single])
    assert chosen == single


def test_unsplit_and_complete_split_ambiguous(tmp_path):
    single = tmp_path / "model.gguf"
    single.write_bytes(b"GGUF")
    shard1 = tmp_path / "model-00001-of-00002.gguf"
    shard2 = tmp_path / "model-00002-of-00002.gguf"
    shard1.write_bytes(b"GGUF")
    shard2.write_bytes(b"GGUF")
    # Two unambiguous models (one unsplit, one complete split) => ambiguous.
    with pytest.raises(ModelStoreError, match="exactly one unambiguous"):
        choose_default_gguf([single, shard1, shard2])


def test_choose_default_gguf_completes_split(tmp_path):
    shard1 = tmp_path / "model-00001-of-00002.gguf"
    shard2 = tmp_path / "model-00002-of-00002.gguf"
    shard1.write_bytes(b"GGUF")
    shard2.write_bytes(b"GGUF")
    chosen = choose_default_gguf([shard1, shard2])
    assert chosen.name == "model-00001-of-00002.gguf"


def test_choose_default_gguf_rejects_incomplete_split(tmp_path):
    shard1 = tmp_path / "model-00001-of-00003.gguf"
    shard1.write_bytes(b"GGUF")
    with pytest.raises(ModelStoreError, match="exactly one unambiguous"):
        choose_default_gguf([shard1])


def test_choose_default_gguf_rejects_mmproj(tmp_path):
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    mmproj.write_bytes(b"GGUF")
    with pytest.raises(ModelStoreError, match="exactly one unambiguous"):
        choose_default_gguf([mmproj])


def test_discover_local_models(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"GGUF")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.gguf").write_bytes(b"GGUF")
    (tmp_path / "not.bin").write_text("x")
    found = discover_local_models(tmp_path)
    assert len(found) == 2
    assert all(p.suffix == ".gguf" for p in found)


def test_discover_local_models_empty_dir(tmp_path):
    assert discover_local_models(tmp_path) == []


def test_split_shard_parses_and_ignores_standalone(tmp_path):
    from kestrel.model_store import split_shard

    shard = tmp_path / "m-00002-of-00007.gguf"
    assert split_shard(shard) == ("m", 2, 7)
    assert split_shard(tmp_path / "plain.gguf") is None
    assert split_shard(tmp_path / "m-2-of-7.gguf") is None


def test_complete_gguf_models_drops_incomplete_split(tmp_path):
    from kestrel.model_store import complete_gguf_models

    single = tmp_path / "single.gguf"
    single.write_bytes(b"GGUF")
    shard1 = tmp_path / "m-00001-of-00003.gguf"
    shard1.write_bytes(b"GGUF")
    assert complete_gguf_models([single, shard1]) == [single]


def test_complete_gguf_models_returns_first_shard(tmp_path):
    from kestrel.model_store import complete_gguf_models

    shard1 = tmp_path / "m-00001-of-00002.gguf"
    shard2 = tmp_path / "m-00002-of-00002.gguf"
    shard1.write_bytes(b"GGUF")
    shard2.write_bytes(b"GGUF")
    assert complete_gguf_models([shard2, shard1]) == [shard1]


def test_model_total_size_sums_sibling_shards(tmp_path):
    from kestrel.model_store import model_total_size

    shard1 = tmp_path / "m-00001-of-00002.gguf"
    shard2 = tmp_path / "m-00002-of-00002.gguf"
    shard1.write_bytes(b"G" * 10)
    shard2.write_bytes(b"G" * 20)
    standalone = tmp_path / "standalone.gguf"
    standalone.write_bytes(b"G" * 5)
    assert model_total_size(shard1) == 30
    assert model_total_size(standalone) == 5
