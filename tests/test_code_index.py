from __future__ import annotations

from kestrel.code_index import CodeIndex, render_evidence


def test_code_index_retrieves_symbols_and_current_source(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    source = workspace / "ledger.py"
    source.write_text(
        "class EventLedger:\n    def append_event(self, stream):\n        return stream.strip()\n",
        encoding="utf-8",
    )
    index = CodeIndex(workspace, tmp_path / "code.sqlite3")

    stats = index.refresh()
    evidence = index.search("append_event stream")

    assert stats["indexed"] == 1
    assert evidence[0].path == "ledger.py"
    assert "EventLedger" in evidence[0].symbols
    assert "append_event" in evidence[0].content


def test_refresh_replaces_stale_chunks_and_removes_deleted_files(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    source = workspace / "service.py"
    source.write_text("def old_handler():\n    return 'oldneedle'\n", encoding="utf-8")
    index = CodeIndex(workspace, tmp_path / "code.sqlite3")
    index.refresh()
    assert index.search("oldneedle")

    source.write_text("def new_handler():\n    return 'newneedle'\n", encoding="utf-8")
    index.refresh()
    assert not index.search("oldneedle")
    assert index.search("newneedle")[0].digest

    source.unlink()
    stats = index.refresh()
    assert stats["removed"] == 1
    assert not index.search("newneedle")


def test_secret_binary_and_oversized_files_are_not_indexed(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / ".env").write_text("API_SECRET=needle\n", encoding="utf-8")
    (workspace / "private.pem").write_text("PRIVATE needle\n", encoding="utf-8")
    (workspace / "credentials.json").write_text('{"token": "credentialneedle"}', encoding="utf-8")
    (workspace / "binary.py").write_bytes(b"prefix\0needle")
    (workspace / "normal.py").write_text("safe_value = 'visible'\n", encoding="utf-8")
    index = CodeIndex(workspace, tmp_path / "code.sqlite3")

    index.refresh()

    assert index.stats()["files"] == 1
    assert not index.search("API_SECRET")
    assert not index.search("PRIVATE")
    assert not index.search("credentialneedle")
    assert index.search("visible")


def test_rendered_evidence_is_bounded_and_source_labelled(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "module.py").write_text(
        "def relevant_symbol():\n" + "    value = 'relevant_symbol'\n" * 100,
        encoding="utf-8",
    )
    index = CodeIndex(workspace, tmp_path / "code.sqlite3")
    index.refresh()

    rendered = render_evidence(index.search("relevant_symbol", k=5), max_chars=240)

    assert len(rendered) <= 240
    assert "module.py:1 digest=" in rendered


def test_rendered_evidence_focuses_lines_and_symbols_on_query(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    symbols = "\n".join(f"def helper_{index}(): pass" for index in range(20))
    target = "\n".join(
        [
            *(f"padding_{index} = {index}" for index in range(45)),
            "def target_handler():",
            "    return 42",
        ]
    )
    (workspace / "module.py").write_text(
        symbols + "\n" + target,
        encoding="utf-8",
    )
    index = CodeIndex(workspace, tmp_path / "code.sqlite3")
    index.refresh()

    rendered = render_evidence(
        index.search("target_handler", k=3),
        max_chars=2_000,
        query="Explain target_handler",
    )

    assert "target_handler" in rendered
    assert "padding_0" not in rendered
    header = rendered.splitlines()[0]
    assert "symbols=target_handler" in header
    assert "helper_0" not in header
