import pytest

from kestrel.gguf.converter import (
    _ROUTER_SUFFIX,
    _resolve_pruning,
    _select_kept_experts,
)

# -----------------------------------------------------------------------------
# kept-expert selection -------------------------------------------------------


def test_select_kept_first_n_without_importance():
    assert _select_kept_experts(256, 8, None) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_select_kept_highest_importance_deterministic():
    importance = list(range(256))  # expert 255 is most important
    assert _select_kept_experts(256, 4, importance) == [252, 253, 254, 255]


def test_select_kept_ties_break_by_index():
    importance = [5, 5, 1, 1]
    assert _select_kept_experts(4, 2, importance) == [0, 1]


def test_select_kept_importance_wrong_length_rejected():
    with pytest.raises(ValueError, match="expected 4"):
        _select_kept_experts(4, 2, [1.0, 2.0])


def test_select_kept_is_sorted_ascending():
    kept = _select_kept_experts(100, 5, importance=[0] * 100)
    assert kept == sorted(kept)


# -----------------------------------------------------------------------------
# pruning resolution ----------------------------------------------------------


def test_resolve_pruning_off():
    assert _resolve_pruning(256, 8, None) == (None, 256)


def test_resolve_pruning_keeps_count_and_indices():
    kept, emitted = _resolve_pruning(256, 8, 16)
    assert emitted == 16
    assert len(kept) == 16
    assert kept == list(range(16))


def test_resolve_pruning_importance_selects_high_value_experts():
    kept, emitted = _resolve_pruning(256, 8, 16, importance=list(range(256)))
    assert emitted == 16
    assert kept == list(range(240, 256))


def test_resolve_pruning_underflows_routing_width_rejected():
    with pytest.raises(ValueError, match="cannot be smaller"):
        _resolve_pruning(256, 8, 4)


def test_resolve_pruning_noop_kept_equal_n_exp_rejected():
    with pytest.raises(ValueError, match="must be smaller"):
        _resolve_pruning(256, 8, 256)


# -----------------------------------------------------------------------------
# router <-> expert consistency (the reviewer-reported class of bug) ----------


def test_router_suffix_only_matches_moe_router():
    assert "mlp.gate.weight".endswith(_ROUTER_SUFFIX)
    assert not "mlp.shared_expert_gate.weight".endswith(_ROUTER_SUFFIX)
    assert not "mlp.gate_proj.weight".endswith(_ROUTER_SUFFIX)


def test_pruned_router_slices_kept_rows_in_emission_order():
    """Kept expert tensors and router rows must be reduced identically.

    ``mlp.gate.weight`` is (n_exp, n_embd); under pruning row ``e`` belongs to
    original expert ``e``. Slicing rows by the ascending kept index list keeps
    router column ``i`` aligned with emitted expert ``i``.
    """
    n_exp, n_embd = 8, 6
    import torch

    kept, emitted = _resolve_pruning(n_exp, 2, 4, importance=[0.1, 9, 9, 9, 0.1, 9, 0.1, 0.1])
    assert len(kept) == emitted == 4

    router = torch.arange(n_exp * n_embd).reshape(n_exp, n_embd)
    sliced = router[kept]
    assert sliced.shape == (emitted, n_embd)
    for e, src in enumerate(kept):
        assert torch.equal(sliced[e], router[src])

    # Router and expert tensors agree on every emitted dimension after pruning.
    assert len(kept) == emitted
    assert list(kept) == sorted(kept)


def test_pruned_router_without_kept_is_identity():
    """Without pruning the router must pass through byte-identical."""
    import torch

    from kestrel.gguf import converter

    conv = object.__new__(converter.NVFP4Converter)
    conv._kept = None
    t = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    assert conv._pruned_router(t, "layers.0.mlp.gate.weight") is t
