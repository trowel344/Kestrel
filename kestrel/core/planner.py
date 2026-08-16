from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

MIB = 1024**2

# Memory-bound decode model. Double-digit decode on an RTX 4060 Laptop is
# exactly the bandwidth ceiling of the ~10B active working set at ~4-6 bits.
DEFAULT_GPU_BANDWIDTH_GB_S = 256.0
DEFAULT_RAM_BANDWIDTH_GB_S = 25.0
# Effective sustained throughput for routed-expert matvecs on the reference
# i7-13620H. Calibrated from the exact Q4 full model at 12 GPU layers:
# 1.39 +/- 0.01 generated tok/s. This is intentionally an end-to-end effective
# rate, not the CPU's peak GEMM throughput.
DEFAULT_CPU_GFLOPS = 10.0
# Full 48-layer Qwen3.5-122B-A10B experts-only Q1 cold-sidecar measurement:
# 3.208 +/- 0.018 tok/s with the cache disabled. The earlier four-layer harness
# scaled to 9.83 tok/s and was invalidated by this complete-model result.
DEFAULT_CPU_Q1_LAYER_TPS = 3.208268 * 48
FLOP_PER_PARAM_PER_TOKEN = 2.0
DRAFT_SPEEDUP = 1.8

# Effective bytes per parameter including block scales, by quantization.
BYTES_PER_PARAM = {
    "q1_0": 0.16,
    "q2_k": 0.35,
    "q3_k": 0.50,
    "nvfp4": 0.55,
    "q4_0": 0.56,
    "q4_k": 0.60,
    "q5_k": 0.75,
    "q6_k": 0.85,
    "q8_0": 1.06,
    "bf16": 2.00,
}

_GPU_BANDWIDTH_TABLE = (
    ("rtx 4060", 256.0),
    ("rtx 4060 ti", 288.0),
    ("rtx 4070", 504.0),
    ("rtx 4080", 716.8),
    ("rtx 4090", 1008.0),
    ("rtx 3080", 760.0),
    ("rtx 3090", 936.0),
    ("rtx 3070", 448.0),
    ("rtx 3060", 360.0),
    ("rtx 2080", 448.0),
    ("gtx 1080", 320.0),
    ("a100", 2039.0),
    ("h100", 3350.0),
)


@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str | None = None
    vram_total_mib: int = 0
    vram_free_mib: int = 0
    ram_available_mib: int = 0
    logical_cpu_count: int = 0
    cpu_gflops: float = DEFAULT_CPU_GFLOPS


@dataclass(frozen=True)
class ModelProfile:
    path: str
    n_layers: int
    n_experts: int
    n_experts_used: int
    hidden_size: int
    expert_ff_size: int
    has_mtp: bool
    file_size_bytes: int = 0
    # GGUF `general.file_type` id (0 when unknown/not a GGUF). Used to anchor
    # the parameter estimate against the file's own size; see estimate_parameters.
    file_type: int = 0


@dataclass(frozen=True)
class RuntimePlan:
    gpu_layers: str = "auto"
    fit: bool = True
    fit_target_mib: int = 1024
    cpu_moe: bool = False
    n_cpu_moe_layers: int | None = None
    context_size: int = 2048
    batch_size: int = 512
    ubatch_size: int = 128
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    use_mtp: bool = False
    moe_cache: str = "auto"
    moe_cache_budget_mib: int = 0
    mmap: bool = True
    threads: int = 0
    # Dedicated CPU thread budget for prompt processing (--threads-batch).
    # 0 means "reuse the decode thread count"; when CPU-MoE is active the
    # planner raises it to the full logical core count because prefill is
    # heavily parallel CPU work for the layers that stay resident.
    threads_batch: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def gpu_bandwidth_gb_s(name: str | None) -> float:
    """Effective HBM/GDDR bandwidth for a GPU, falling back to a conservative
    default so small discrete GPUs are not over-promised."""
    if not name:
        return DEFAULT_GPU_BANDWIDTH_GB_S
    lowered = name.lower()
    # Match the longest (most specific) key first so a model like "rtx 4060 ti"
    # is not captured by the shorter substring "rtx 4060".
    matches = [(len(key), value) for key, value in _GPU_BANDWIDTH_TABLE if key in lowered]
    if matches:
        return max(matches)[1]
    return DEFAULT_GPU_BANDWIDTH_GB_S


# Nominal bits per weight for GGUF `general.file_type` ids (llama.cpp ggml
# block types). Missing ids (e.g. unquantized/converted files that omit or
# write -1) simply disable the file-size anchor below.
FILE_TYPE_BITS = {
    0: 32.0,  # F32
    1: 16.0,  # F16
    2: 4.5625,  # Q4_0
    3: 5.5625,  # Q4_1
    5: 4.5625,  # Q4_2
    6: 5.5625,  # Q4_3
    7: 8.5,  # Q8_0
    8: 5.5625,  # Q5_0
    9: 6.5625,  # Q5_1
    11: 9.5,  # Q8_1
    12: 3.1875,  # Q2_K
    13: 4.15625,  # Q3_K
    14: 5.4375,  # Q4_K
    15: 6.4375,  # Q5_K
    16: 7.4375,  # Q6_K
    17: 9.125,  # Q8_K
    18: 2.0625,  # IQ2_XXS
    19: 2.3125,  # IQ2_XS
    20: 2.4375,  # IQ3_XXS
    21: 1.5625,  # IQ1_S
    22: 4.5,  # IQ4_NL
    23: 3.4375,  # IQ3_S
    24: 2.5,  # IQ2_S
    30: 4.25,  # IQ4_XS
    31: 1.75,  # IQ1_M
    32: 8.0,  # I8
}


def estimate_parameters(model: ModelProfile) -> dict:
    """Approximate dense and MoE parameter counts from the architecture.

    Dense per layer uses the four attention projections; each expert is the
    three (up/gate/down) hidden<->FF matrices. Exact counts vary by backend,
    so this is a close, explainable estimate used only for rate prediction.

    When ``general.file_type`` is known, the FFN slice is rescaled so the
    estimated total matches the file's own size. Some hybrid-MoE conversions
    (e.g. a 30B-A3B class model) advertise a ``feed_forward_length`` that does
    not correspond to the tensor payloads -- only a fraction of the layers
    carry experts -- which would otherwise inflate the total to a multi-TB
    phantom and make every prediction absurdly slow. The active set is scaled
    the same way so the routed-expert working set stays consistent with the
    file.
    """
    attention_per_layer = 4 * model.hidden_size * model.hidden_size
    expert_params = 3 * model.hidden_size * model.expert_ff_size
    # A dense transformer still has gate/up/down FFN matrices. The previous
    # estimator omitted them whenever expert_count was zero, making 4B dense
    # models look like ~100M models and producing absurd throughput claims.
    dense_ffn_per_layer = expert_params if model.n_experts == 0 else 0
    dense_per_layer = attention_per_layer + dense_ffn_per_layer
    total = model.n_layers * (dense_per_layer + model.n_experts * expert_params)
    active = model.n_layers * (dense_per_layer + model.n_experts_used * expert_params)
    bits = FILE_TYPE_BITS.get(model.file_type)
    if bits and model.file_size_bytes > 0:
        anchor = model.file_size_bytes * 8 / bits
        dense_total = model.n_layers * dense_per_layer
        ffn_total = total - dense_total
        if ffn_total > 0 and dense_total < anchor:
            scale = (anchor - dense_total) / ffn_total
            # Within ~30% the advertised structure is credible; outside it the
            # metadata is almost certainly inconsistent with the payload, so
            # rescale the FFN slice (and thus both totals) to the file.
            if not 0.7 <= scale <= 1.4:
                expert_params = expert_params * scale
                total = model.n_layers * (dense_per_layer + model.n_experts * expert_params)
                active = model.n_layers * (dense_per_layer + model.n_experts_used * expert_params)
    return {
        "dense_per_layer": dense_per_layer,
        "expert_params": expert_params,
        "total_params": total,
        "active_params": active,
    }


def effective_bytes_per_param(model: ModelProfile, params: dict, quant: str) -> float:
    """Use the file's own density when available; otherwise a quant estimate."""
    if model.file_size_bytes and params.get("total_params"):
        from_file = model.file_size_bytes / params["total_params"]
        if 0.1 <= from_file <= 2.0:
            return from_file
    return BYTES_PER_PARAM.get(quant, 0.6)


def predict_decode_tokens_per_second(
    model: ModelProfile,
    hardware: HardwareProfile,
    *,
    cpu_moe: bool,
    moe_cache_budget_mib: int = 0,
    gpu_layers_offloaded: int = 0,
    draft: bool = False,
    quant: str = "q4_0",
    cpu_expert_quant: str | None = None,
) -> float:
    """Predict memory-bound decode throughput.

    Decode reads the active working set (dense layers plus the routed experts)
    every token. When the active set stays in VRAM the rate is the GPU
    bandwidth divided by that footprint. CPU-MoE is bounded by the CPU expert
    matmuls: measured on the reference laptop the expert-cache budget does not
    offload that work, so the rate is flat regardless of cache size.
    Speculative decoding hides most expert lookups behind accepted draft tokens.
    """
    params = estimate_parameters(model)
    if not params["active_params"]:
        return 0.0
    bpp = effective_bytes_per_param(model, params, quant)
    gpu_bw = gpu_bandwidth_gb_s(hardware.gpu_name)
    ram_bw = DEFAULT_RAM_BANDWIDTH_GB_S

    active_bytes = params["active_params"] * bpp / 1024**3
    dense_bytes = model.n_layers * params["dense_per_layer"] * bpp / 1024**3
    expert_bpp = BYTES_PER_PARAM.get(cpu_expert_quant, bpp) if cpu_expert_quant else bpp
    expert_bytes = model.n_layers * model.n_experts_used * params["expert_params"] * expert_bpp / 1024**3

    if not cpu_moe:
        tps = gpu_bw / active_bytes
    else:
        offload_frac = min(1.0, max(0.0, gpu_layers_offloaded / max(1, model.n_layers)))
        hit_frac = 0.0
        if moe_cache_budget_mib > 0 and expert_bytes > 0:
            hit_frac = min(1.0, (moe_cache_budget_mib / 1024) / expert_bytes)
        dense_on_gpu = dense_bytes * offload_frac
        expert_on_gpu = expert_bytes * hit_frac
        streamed = (dense_bytes - dense_on_gpu) + (expert_bytes - expert_on_gpu)
        gpu_time = (dense_on_gpu + expert_on_gpu) / gpu_bw
        cpu_time = streamed / ram_bw
        tps = 1.0 / max(1e-9, gpu_time + cpu_time)

        # All routed experts are computed on the CPU in CPU-MoE mode; the
        # expert cache avoids weight re-uploads but not the matmuls, so it does
        # not relax this compute bound.
        cpu_flops = model.n_layers * model.n_experts_used * params["expert_params"] * FLOP_PER_PARAM_PER_TOKEN
        cpu_tps = (hardware.cpu_gflops * 1e9) / max(1e-9, cpu_flops)
        if cpu_expert_quant == "q1_0":
            cpu_tps = DEFAULT_CPU_Q1_LAYER_TPS / max(1, model.n_layers)
        tps = min(tps, cpu_tps)

    if draft:
        tps *= DRAFT_SPEEDUP
    return round(tps, 2)


def plan_runtime(
    model: ModelProfile,
    hardware: HardwareProfile,
    *,
    context_size: int = 2048,
    requested_gpu_layers: str = "auto",
    requested_cpu_moe: bool | None = None,
    mode: str = "balanced",
) -> RuntimePlan:
    """Create a conservative llama.cpp plan.

    Kestrel uses a measured fixed placement only for a recognized model and
    hardware class. Other layouts keep conservative defaults and llama.cpp's
    allocation fitter plus a safety margin.

    ``mode`` selects from ``balanced`` (current, memory-aware), ``quality``
    (extra conservative and slower but maximally stable), or ``speed``
    (explicit experimental throughput bias). Only ``balanced`` is ever chosen
    automatically; the other two require an explicit user request so an
    unstable faster profile is never silently selected.
    """
    total_vram = max(0, hardware.vram_total_mib)
    free_vram = max(0, hardware.vram_free_mib)

    fit_target = _fit_target_mib(total_vram)

    usable_vram_bytes = max(0, free_vram - fit_target) * MIB
    model_is_larger_than_vram = bool(model.file_size_bytes and model.file_size_bytes > usable_vram_bytes)
    cpu_moe = model_is_larger_than_vram if requested_cpu_moe is None else requested_cpu_moe
    threads = _tune_threads(cpu_moe, hardware.logical_cpu_count)
    threads_batch = _tune_threads_batch(cpu_moe, hardware.logical_cpu_count)

    full_dense_fit = _can_full_dense_offload(
        model,
        hardware,
        cpu_moe=cpu_moe,
        fit_target_mib=fit_target,
    )
    batch_size, ubatch_size = _select_batch_sizes(
        total_vram,
        cpu_moe=cpu_moe,
        verified=full_dense_fit,
    )
    gpu_layers = _resolve_verified_placement(
        requested_gpu_layers,
        cpu_moe=cpu_moe,
        n_experts=model.n_experts,
        total_vram=total_vram,
        verified=full_dense_fit,
        n_layers=model.n_layers,
    )
    moe_cache, moe_cache_budget_mib = _select_moe_cache(cpu_moe, model.n_experts)

    return _apply_mode(
        RuntimePlan(
            gpu_layers=gpu_layers,
            fit=True,
            fit_target_mib=fit_target,
            cpu_moe=cpu_moe and model.n_experts > 0,
            n_cpu_moe_layers=None,
            context_size=max(512, context_size),
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
            # On 8 GiB hardware, an oversized CPU-MoE target plus its MTP context
            # either OOMs or requires an all-CPU profile that is slower than the
            # non-speculative baseline. Do not enable it automatically there.
            use_mtp=model.has_mtp and not (cpu_moe and total_vram and total_vram <= 8192),
            moe_cache=moe_cache,
            moe_cache_budget_mib=moe_cache_budget_mib,
            mmap=True,
            threads=threads,
            threads_batch=threads_batch,
        ),
        mode=mode,
        model=model,
        ram_available_mib=hardware.ram_available_mib,
    )


def _fit_target_mib(total_vram: int) -> int:
    """Keep room for the display server, CUDA context, and allocation
    fragmentation. Small GPUs need a proportionally larger guard."""
    if total_vram:
        return max(1024, min(3072, int(total_vram * 0.18)))
    return 1024


def _tune_threads(cpu_moe: bool, logical_cpu_count: int) -> int:
    if not cpu_moe or not logical_cpu_count:
        return 0
    # CPU-MoE decode is memory-bandwidth bound. Leaving two logical CPUs
    # for CUDA/runtime work and avoiding very large thread teams performed
    # best in the real-model sweep (14 threads on a 16-thread hybrid CPU).
    return min(14, max(1, logical_cpu_count - 2))


def _tune_threads_batch(cpu_moe: bool, logical_cpu_count: int) -> int:
    """Dedicated prompt-processing thread budget for CPU-MoE plans.

    Prefill over the CPU-resident layers is a parallel GEMM/dequant workload
    that saturates more threads than memory-bound decode. Capped at 16 to avoid
    oversubscribing hybrid parts, and 0 (reuse decode threads) when no CPU work
    is expected or the CPU count is unknown.
    """
    if not cpu_moe or not logical_cpu_count:
        return 0
    return min(16, max(1, logical_cpu_count))


def _can_full_dense_offload(
    model: ModelProfile,
    hardware: HardwareProfile,
    *,
    cpu_moe: bool,
    fit_target_mib: int,
) -> bool:
    """Conservatively estimate whether every non-expert layer fits on GPU.

    This is deliberately architecture-generic. The estimate uses Q4 density
    even when the whole-file density appears lower (grouped MoE metadata can
    make parameter estimates misleading), doubles the attention footprint for
    unmodelled routers/shared/SSM tensors, and reserves another GiB for graphs.
    Aggressive expert and micro-batch placement belongs to measured profiles.
    """
    if not cpu_moe or model.n_experts <= 0 or not hardware.vram_free_mib:
        return False
    params = estimate_parameters(model)
    dense_bytes = model.n_layers * params["dense_per_layer"] * max(BYTES_PER_PARAM["q4_k"], 0.6)
    conservative_dense_mib = (dense_bytes * 2.0) / MIB
    available = hardware.vram_free_mib - fit_target_mib - 1024
    return available > 0 and conservative_dense_mib <= available


def _select_batch_sizes(
    total_vram: int,
    *,
    cpu_moe: bool,
    verified: bool,
) -> tuple[int, int]:
    # Smaller physical batches reduce temporary CUDA allocations. Logical
    # batches can remain larger for prompt throughput.
    if total_vram and total_vram <= 8192:
        batch_size, ubatch_size = 512, 128
    elif total_vram and total_vram <= 16384:
        batch_size, ubatch_size = 1024, 256
    else:
        batch_size, ubatch_size = 2048, 512

    if cpu_moe and total_vram and total_vram <= 8192 and verified:
        # Keep temporary CUDA allocations bounded for the verified 8 GiB
        # hybrid placement. Routed experts remain on CPU, so all dense layers
        # and the output tensor can still fit; the smaller physical batch
        # leaves the fitter room to account for context and graph buffers.
        batch_size, ubatch_size = 256, 64
    return batch_size, ubatch_size


def _resolve_verified_placement(
    requested_gpu_layers: str,
    *,
    cpu_moe: bool,
    n_experts: int,
    total_vram: int,
    verified: bool,
    n_layers: int,
) -> str:
    if not (requested_gpu_layers == "auto" and cpu_moe and n_experts > 0 and total_vram and total_vram <= 8192):
        return requested_gpu_layers
    # llama.cpp's fitter currently accounts poorly for some mixed
    # CPU-MoE/CUDA layouts. Use a full dense-layer placement when the generic
    # conservative footprint estimate fits; retain the four-layer fallback
    # when it does not.
    # Dense models are left to llama.cpp's own fitter so they offload as
    # many layers as fit instead of being pinned to four CPU layers.
    if verified:
        # n_layers + 1 requests every transformer block plus the output tensor.
        # Partial expert placement and larger prefill batches are never guessed
        # here; adaptive tuning must measure and persist those overrides.
        return str(max(0, n_layers) + 1)
    return str(min(4, max(0, n_layers)))


def _select_moe_cache(cpu_moe: bool, n_experts: int) -> tuple[str, int]:
    # The current cache hook launches CUDA work from CPU mul_mat_id and then
    # synchronizes its result back to the CPU graph. It is slower on both the
    # compact harness and the full calibrated 122B artifact, and enabling it
    # also disables CPU weight repacking. Force it off for CPU-MoE plans;
    # explicit --moe-cache budgets can still opt into controlled experiments.
    if cpu_moe and n_experts > 0:
        return "off", 0
    return "auto", 0


def _apply_mode(
    plan: RuntimePlan,
    *,
    mode: str,
    model: ModelProfile,
    ram_available_mib: int,
) -> RuntimePlan:
    """Apply the requested placement mode on top of the balanced plan.

    ``balanced`` is the identity: it always reflects the memory-aware defaults
    above. The other two are explicit opt-ins and never the automatic choice.
    """
    if mode not in ("quality", "speed"):
        return plan
    if mode == "quality":
        # Maximally conservative: disable speculative decoding and shrink the
        # physical batch so temporary allocations stay small regardless of RAM.
        return replace(
            plan,
            use_mtp=False,
            batch_size=min(plan.batch_size, 256),
            ubatch_size=min(plan.ubatch_size, 64),
            moe_cache="off",
        )
    # speed: explicit experimental throughput bias. Never auto-selected.
    if not model.has_mtp:
        # No speculation available: raise the physical batch for CPU-MoE
        # (the memory-saving default is smaller) when RAM has real headroom.
        # Never lower an already-larger batch.
        if ram_available_mib >= 4096 and plan.ubatch_size < 128:
            return replace(plan, batch_size=512, ubatch_size=128)
        return plan
    # Otherwise relax the 8 GiB auto-disable of MTP, but only when there is
    # real RAM headroom to absorb the extra context (opting in is explicit).
    if not plan.use_mtp and ram_available_mib >= 4096:
        return replace(plan, use_mtp=True)
    return plan


def model_file_size(path: str) -> int:
    candidate = Path(path)
    return candidate.stat().st_size if candidate.is_file() else 0
