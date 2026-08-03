#!/usr/bin/env python3
import sys
import time
sys.path.insert(0, "/home/cleanerbox/kestrel")

import torch
from kestrel.verify.colibri_engine import ColibriEngine
from kestrel.verify.expert_cache import ExpertCache
from kestrel.verify.kv_cache import KVCacheManager


def print_header(title):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def benchmark_single_layer(engine: ColibriEngine, layer_idx: int = 0, seq_len: int = 1, n_runs: int = 100):
    layer = engine.layers[layer_idx]
    hidden = torch.randn(1, seq_len, engine.hidden_size, device=engine.device)
    pos_ids = torch.arange(seq_len, device=engine.device).unsqueeze(0)
    pos_emb = engine.rotary(hidden, pos_ids)

    old_exp = layer.mlp.experts
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    full_exp = Qwen3_5MoeExperts(engine.config)
    with torch.no_grad():
        full_exp.gate_up_proj.data = torch.randn(256, 1024, 2048, dtype=engine.dtype, device=engine.device)
        full_exp.down_proj.data = torch.randn(256, 2048, 512, dtype=engine.dtype, device=engine.device)
    layer.mlp.experts = full_exp

    for _ in range(10):
        layer(hidden_states=hidden, position_embeddings=pos_emb)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        layer(hidden_states=hidden, position_embeddings=pos_emb)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) * 1000 / n_runs

    layer.mlp.experts = old_exp
    return {"ms_per_layer": avg_ms, "seq_len": seq_len}


def benchmark_pcie_single_layer(engine: ColibriEngine):
    print_header("PCIe Bandwidth (Single Layer)")
    total_bytes = (256 * 1024 * 2048 * 2 + 256 * 2048 * 512 * 2)
    print(f"  Expert weights per layer: {total_bytes/1024**3:.2f} GB")

    engine._expert_cache.clear()

    # Load full expert tensor to CPU (measure disk→CPU if not cached)
    t0 = time.perf_counter()
    gate_up = engine._load_tensor("model.language_model.layers.0.mlp.experts.gate_up_proj")
    down = engine._load_tensor("model.language_model.layers.0.mlp.experts.down_proj")
    load_cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"  Load from disk→CPU: {load_cpu_ms:.0f}ms")

    # Measure CPU→GPU transfer (full tensor)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        g = gate_up.to(device=engine.device, non_blocking=True)
        d = down.to(device=engine.device, non_blocking=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        del g, d

    avg_gpu = sum(times) / len(times)
    bw = total_bytes / (avg_gpu / 1000) / 1024**3
    print(f"  CPU→GPU: {[f'{t:.0f}' for t in times]}")
    print(f"  Avg: {avg_gpu:.0f}ms @ {bw:.1f} GB/s")

    # Measure GPU→CPU transfer
    gate_up_gpu = gate_up.to(device=engine.device)
    down_gpu = down.to(device=engine.device)
    torch.cuda.synchronize()
    times_out = []
    for _ in range(3):
        t0 = time.perf_counter()
        _gc = gate_up_gpu.to(device='cpu', non_blocking=True)
        _dc = down_gpu.to(device='cpu', non_blocking=True)
        torch.cuda.synchronize()
        times_out.append((time.perf_counter() - t0) * 1000)
        del _gc, _dc

    avg_gpu_out = sum(times_out) / len(times_out)
    bw_out = total_bytes / (avg_gpu_out / 1000) / 1024**3
    print(f"  GPU→CPU: {[f'{t:.0f}' for t in times_out]}")
    print(f"  Avg: {avg_gpu_out:.0f}ms @ {bw_out:.1f} GB/s")

    del gate_up_gpu, down_gpu, gate_up, down

    # Now measure just the selected 8 experts (what colibri actually transfers)
    engine._expert_cache.clear()
    exp_indices = torch.tensor(range(8), device=engine.device)
    gate_up_cpu = engine._load_tensor("model.language_model.layers.0.mlp.experts.gate_up_proj")
    down_cpu = engine._load_tensor("model.language_model.layers.0.mlp.experts.down_proj")
    selected_bytes = (8 * 1024 * 2048 * 2 + 8 * 2048 * 512 * 2)

    t0 = time.perf_counter()
    sel_g = gate_up_cpu[exp_indices.cpu()].to(device=engine.device)
    sel_d = down_cpu[exp_indices.cpu()].to(device=engine.device)
    torch.cuda.synchronize()
    sel_ms = (time.perf_counter() - t0) * 1000
    bw_sel = selected_bytes / (sel_ms / 1000) / 1024**3
    print(f"  Selected 8 experts ({selected_bytes/1024**2:.0f} MB): {sel_ms:.1f}ms @ {bw_sel:.1f} GB/s")

    del sel_g, sel_d, gate_up_cpu, down_cpu

    return {
        "load_ms": avg_gpu,
        "unload_ms": avg_gpu_out,
        "bw_gbps": bw,
        "bw_out_gbps": bw_out,
        "selected_8_ms": sel_ms,
        "size_gb": total_bytes/1024**3,
        "disk_to_cpu_ms": load_cpu_ms,
    }


def benchmark_expert_cache():
    print_header("Expert Cache Benchmark")
    cache = ExpertCache(model_path="", device="cuda", vram_budget_gb=4.0)
    n_accesses = 40
    total_time = 0
    for i in range(n_accesses):
        t0 = time.perf_counter()
        cache.get_expert_weights(i % 40)
        total_time += (time.perf_counter() - t0) * 1000
    s = cache.summary()
    print(f"  Accesses: {n_accesses}")
    print(f"  Hit rate: {s['hit_rate']}")
    print(f"  Cache: {s['resident_layers']} layers ({s['cache_used_gb']:.3f} GB / {s['cache_budget_gb']} GB)")
    print(f"  Avg load time: {total_time/n_accesses:.1f}ms")
    return s


def benchmark_kv_cache():
    print_header("KV Cache Benchmark")
    kv = KVCacheManager(num_full_attn_layers=10, num_kv_heads=2, head_dim=256, max_seq_len=32768)
    print(f"  KV dim per layer: {kv.kv_dim}")
    print(f"  Full attention layers: {kv.num_layers}")
    print(f"  Max seq_len: {kv.max_seq_len}")
    s = kv.summary()
    print(f"  Active: {s['layers_active']}")
    print(f"  VRAM: {s['vram_gb']} GB")
    return s


def compute_model_estimate(results):
    layer_decode = results["layer_decode"]
    layer_prefill = results["layer_prefill"]
    pcie = results["pcie"]

    decode_per_layer = layer_decode["ms_per_layer"]
    prefill_per_layer = layer_prefill["ms_per_layer"]
    pcie_load = pcie["selected_8_ms"]

    num_layers = 40
    total_decode = num_layers * (decode_per_layer + pcie_load)
    tok_s = 1000 / total_decode

    prefill_seq = layer_prefill["seq_len"]
    total_prefill = num_layers * (prefill_per_layer + pcie_load)
    prefill_tok_s = prefill_seq / (total_prefill / 1000)

    # Double-buffered PCIe: load next layer while computing current
    # First layer: load + compute. Subsequent: max(load, compute) per layer
    first_layer_ms = pcie_load + decode_per_layer
    per_layer_ms = max(pcie_load, decode_per_layer)
    total_decode_opt = first_layer_ms + (num_layers - 1) * per_layer_ms
    tok_s_opt = 1000 / total_decode_opt

    # Async CPU→GPU for selected experts (no need to load full 256)
    # Overlapped time = max(selected_8_ms, compute_per_layer)
    async_layer_ms = max(pcie.get('selected_8_ms', pcie_load), decode_per_layer)
    total_decode_async = pcie.get('selected_8_ms', pcie_load) + decode_per_layer + (num_layers - 1) * async_layer_ms
    tok_s_async = 1000 / total_decode_async

    return {
        "decode_tok_s": tok_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s_opt": tok_s_opt,
        "decode_tok_s_async": tok_s_async,
        "decode_per_layer_ms": decode_per_layer,
        "prefill_per_layer_ms": prefill_per_layer,
        "pcie_per_layer_ms": pcie_load,
        "num_layers": num_layers,
    }


def main():
    print("=" * 65)
    print("  KESTREL :: Full Pipeline Benchmark")
    print("  Qwen3.6-35B-A3B on RTX 4060 (8 GB VRAM)")
    print("=" * 65)

    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {vram_total:.1f} GB total")

    print_header("1. Building Colibri Engine")
    engine = ColibriEngine(model_name="Qwen/Qwen3.6-35B-A3B", device="cuda")
    vram_used = torch.cuda.memory_allocated(0) / 1e9
    print(f"  VRAM after build: {vram_used:.2f} GB ({(vram_total-vram_used):.2f} GB free)")

    results = {}
    results["kv_cache"] = benchmark_kv_cache()
    results["expert_cache"] = benchmark_expert_cache()
    results["pcie"] = benchmark_pcie_single_layer(engine)

    print_header("2. Single Layer Compute (experts pre-loaded on GPU)")
    print("  Testing layer 0 (linear_attention)...")
    r = benchmark_single_layer(engine, layer_idx=0, seq_len=1, n_runs=100)
    results["layer_decode"] = r
    print(f"  Decode (seq=1): {r['ms_per_layer']:.3f}ms per layer")

    r = benchmark_single_layer(engine, layer_idx=0, seq_len=128, n_runs=10)
    results["layer_prefill"] = r
    print(f"  Prefill (seq=128): {r['ms_per_layer']:.3f}ms per layer")

    print_header("3. Full Model Estimate")
    est = compute_model_estimate(results)
    results["estimate"] = est
    decode_ms_per_layer = est['decode_per_layer_ms'] + est['pcie_per_layer_ms']
    decode_total_ms = est['num_layers'] * decode_ms_per_layer
    print(f"  Decode: {est['decode_per_layer_ms']:.3f}ms/layer compute + {est['pcie_per_layer_ms']:.0f}ms/layer PCIe")
    print(f"    = {decode_ms_per_layer:.0f}ms/layer")
    print(f"    = {decode_total_ms:.0f}ms/tok  → {est['decode_tok_s']:.1f} tok/s")
    print(f"  With double-buffered PCIe: {est['decode_tok_s_opt']:.1f} tok/s")
    print(f"  With async CPU→GPU copy:   {est['decode_tok_s_async']:.1f} tok/s")
    print(f"  Prefill: {est['prefill_tok_s']:.0f} tok/s (seq=128)")

    pcie_selected = results['pcie']['selected_8_ms']

    print_header("4. What-If: INT4 (4x smaller experts)")
    pcie_4x = max(pcie_selected / 4, 0.1)
    decode_4x = est["decode_per_layer_ms"] / 4
    decode_4x_total = pcie_4x + decode_4x + 39 * max(pcie_4x, decode_4x)
    print(f"  PCIe: {pcie_selected:.1f}ms → {pcie_4x:.1f}ms per layer")
    print(f"  Compute: {est['decode_per_layer_ms']:.3f}ms → {decode_4x:.3f}ms per layer")
    print(f"  Async estimate: {decode_4x_total:.0f}ms/tok ({1000/decode_4x_total:.1f} tok/s)")

    print_header("FINAL RESULTS")
    p = results["pcie"]
    print(f"  Dense weights: {vram_used:.2f} GB in VRAM")
    print(f"  Experts per layer: {p['size_gb']:.2f} GB on disk")
    print(f"  PCIe bandwidth: {p['bw_gbps']:.1f} GB/s in, {p['bw_out_gbps']:.1f} GB/s out")
    print(f"\n  {'Mode':30s}  {'tok/s':>8s}  {'ms/tok':>8s}")
    print(f"  {'-'*48}")
    print(f"  {'BF16 decode (real, sequential)':30s}  {est['decode_tok_s']:>8.1f}  {decode_total_ms:>8.0f}")
    print(f"  {'BF16 decode (double-buffered)':30s}  {est['decode_tok_s_opt']:>8.1f}  {'' :>8s}")
    print(f"  {'BF16 decode (async load)':30s}  {est['decode_tok_s_async']:>8.1f}  {'' :>8s}")
    print(f"  {'INT4 decode (async, projected)':30s}  {1000/decode_4x_total:>8.1f}  {decode_4x_total:>8.0f}")

    vram_final = torch.cuda.memory_allocated(0) / 1e9
    print(f"\n  Peak VRAM: {vram_final:.2f} GB")
    print(f"  All benchmarks completed.")

    return results


if __name__ == "__main__":
    main()
