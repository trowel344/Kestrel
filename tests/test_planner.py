import unittest

from kestrel.core.planner import (
    HardwareProfile,
    ModelProfile,
    effective_bytes_per_param,
    estimate_parameters,
    gpu_bandwidth_gb_s,
    plan_runtime,
    predict_decode_tokens_per_second,
)


def model(size_gib=40, experts=256, mtp=True):
    return ModelProfile(
        path="/model.gguf",
        n_layers=48,
        n_experts=experts,
        n_experts_used=8,
        hidden_size=3072,
        expert_ff_size=1024,
        has_mtp=mtp,
        file_size_bytes=int(size_gib * 1024**3),
    )


class RuntimePlannerTests(unittest.TestCase):
    def test_large_moe_uses_cpu_moe_and_memory_saving_defaults(self):
        plan = plan_runtime(
            model(),
            HardwareProfile("RTX 4060", 8188, 7600, 10000, 16),
        )
        self.assertEqual(plan.gpu_layers, "12")
        self.assertTrue(plan.fit)
        self.assertTrue(plan.cpu_moe)
        self.assertGreaterEqual(plan.fit_target_mib, 1024)
        self.assertEqual(plan.batch_size, 256)
        self.assertEqual(plan.ubatch_size, 64)
        self.assertEqual(plan.cache_type_k, "q8_0")
        self.assertTrue(plan.mmap)
        self.assertFalse(plan.use_mtp)
        self.assertEqual(plan.threads, 14)

    def test_small_model_does_not_force_cpu_moe(self):
        plan = plan_runtime(
            model(size_gib=2),
            HardwareProfile("RTX 4060", 8188, 7600, 10000),
        )
        self.assertFalse(plan.cpu_moe)
        self.assertTrue(plan.use_mtp)

    def test_explicit_cpu_moe_override_is_respected(self):
        hardware = HardwareProfile("RTX 4060", 8188, 7600, 10000)
        self.assertFalse(
            plan_runtime(model(), hardware, requested_cpu_moe=False).cpu_moe
        )
        self.assertTrue(
            plan_runtime(model(size_gib=2), hardware, requested_cpu_moe=True).cpu_moe
        )

    def test_explicit_gpu_layer_override_is_respected(self):
        plan = plan_runtime(
            model(),
            HardwareProfile("RTX 4060", 8188, 7600, 10000),
            requested_gpu_layers="2",
        )
        self.assertEqual(plan.gpu_layers, "2")

    def test_unknown_large_moe_keeps_conservative_layer_fallback(self):
        unknown = ModelProfile(
            path="/model.gguf",
            n_layers=40,
            n_experts=64,
            n_experts_used=4,
            hidden_size=4096,
            expert_ff_size=2048,
            has_mtp=False,
            file_size_bytes=40 * 1024**3,
        )
        plan = plan_runtime(
            unknown,
            HardwareProfile("RTX 4060", 8188, 7600, 10000, 16),
        )
        self.assertEqual(plan.gpu_layers, "4")

    def test_cpu_moe_plan_does_not_force_synchronous_expert_cache(self):
        plan = plan_runtime(
            model(),
            HardwareProfile("RTX 4060", 8188, 7600, 10000, 16),
        )
        self.assertEqual(plan.moe_cache, "off")
        self.assertEqual(plan.moe_cache_budget_mib, 0)

    def test_non_cpu_moe_plan_leaves_cache_auto(self):
        plan = plan_runtime(
            model(size_gib=2),
            HardwareProfile("RTX 4060", 8188, 7600, 10000),
        )
        self.assertEqual(plan.moe_cache, "auto")
        self.assertEqual(plan.moe_cache_budget_mib, 0)


class RatePredictionTests(unittest.TestCase):
    hardware = HardwareProfile(
        "NVIDIA GeForce RTX 4060 Laptop GPU", 8188, 7745, 11000, 16
    )

    def test_cpu_moe_prediction_is_compute_capped_regardless_of_cache(self):
        # Measured reality: the expert cache does not offload CPU matmuls, so
        # decode stays flat on the CPU bound no matter the budget.
        base = predict_decode_tokens_per_second(
            model(), self.hardware, cpu_moe=True, gpu_layers_offloaded=12
        )
        cached = predict_decode_tokens_per_second(
            model(),
            self.hardware,
            cpu_moe=True,
            moe_cache_budget_mib=2048,
            gpu_layers_offloaded=12,
        )
        self.assertEqual(cached, base)

    def test_prediction_increases_with_cache_when_not_cpu_capped(self):
        fast_cpu = HardwareProfile(
            "NVIDIA GeForce RTX 4060 Laptop GPU", 8188, 7745, 11000, 16,
            cpu_gflops=1e6,
        )
        base = predict_decode_tokens_per_second(
            model(), fast_cpu, cpu_moe=True, gpu_layers_offloaded=12
        )
        cached = predict_decode_tokens_per_second(
            model(),
            fast_cpu,
            cpu_moe=True,
            moe_cache_budget_mib=2048,
            gpu_layers_offloaded=12,
        )
        self.assertGreater(cached, base)

    def test_q1_cold_sidecar_uses_measured_fallback_ceiling(self):
        q4 = predict_decode_tokens_per_second(
            model(), self.hardware, cpu_moe=True, gpu_layers_offloaded=12
        )
        q1 = predict_decode_tokens_per_second(
            model(),
            self.hardware,
            cpu_moe=True,
            gpu_layers_offloaded=12,
            cpu_expert_quant="q1_0",
        )
        self.assertGreater(q1, q4)
        self.assertAlmostEqual(q1, 3.21, places=2)

    def test_fully_resident_plan_is_faster_than_cpu_streaming(self):
        resident = predict_decode_tokens_per_second(
            model(), self.hardware, cpu_moe=False
        )
        streaming = predict_decode_tokens_per_second(
            model(), self.hardware, cpu_moe=True
        )
        self.assertGreater(resident, streaming)

    def test_draft_decoding_raises_prediction(self):
        plain = predict_decode_tokens_per_second(
            model(), self.hardware, cpu_moe=True, gpu_layers_offloaded=12
        )
        drafted = predict_decode_tokens_per_second(
            model(),
            self.hardware,
            cpu_moe=True,
            gpu_layers_offloaded=12,
            draft=True,
        )
        self.assertGreater(drafted, plain)

    def test_bandwidth_lookup_orders_gpus(self):
        self.assertGreater(
            gpu_bandwidth_gb_s("NVIDIA GeForce RTX 4090"),
            gpu_bandwidth_gb_s("NVIDIA GeForce RTX 4060 Laptop"),
        )

    def test_bytes_per_param_uses_file_density(self):
        params = estimate_parameters(model(size_gib=40))
        self.assertAlmostEqual(
            effective_bytes_per_param(model(size_gib=40), params, "q4_0"),
            40 * 1024**3 / params["total_params"],
            places=6,
        )

    def test_zero_architecture_gives_zero_prediction(self):
        empty = ModelProfile(
            path="/x.gguf", n_layers=0, n_experts=0, n_experts_used=0,
            hidden_size=0, expert_ff_size=0, has_mtp=False,
        )
        self.assertEqual(
            predict_decode_tokens_per_second(empty, self.hardware, cpu_moe=False),
            0.0,
        )

    def test_balanced_mode_is_identity(self):
        hw = HardwareProfile("RTX 4060", 8188, 7600, 10000, 16)
        base = plan_runtime(model(), hw, mode="balanced")
        default = plan_runtime(model(), hw)
        self.assertEqual(base.as_dict(), default.as_dict())

    def test_quality_mode_forces_conservative_layout(self):
        hw = HardwareProfile("RTX 4090", 24576, 23000, 50000, 16)
        plan = plan_runtime(model(), hw, mode="quality")
        self.assertFalse(plan.use_mtp)
        self.assertLessEqual(plan.batch_size, 256)
        self.assertLessEqual(plan.ubatch_size, 64)
        self.assertEqual(plan.moe_cache, "off")

    def test_speed_mode_relaxes_8giB_mtp_disable_with_ram_headroom(self):
        # Small 8 GiB GPU + same model auto-disables MTP under the default
        # plan; an explicit speed target may re-enable it when RAM is ample.
        hw = HardwareProfile("RTX 4060", 8188, 7600, 10000, 16)
        self.assertFalse(plan_runtime(model(), hw).use_mtp)
        speed = plan_runtime(model(), hw, mode="speed")
        self.assertTrue(speed.use_mtp)

    def test_speed_respects_ram_floor(self):
        # With MTP enabled but little free RAM, speed must not force MTP on.
        hw = HardwareProfile("RTX 4060", 8188, 7600, 2048, 16)
        plan = plan_runtime(model(), hw, mode="speed")
        self.assertFalse(plan.use_mtp)

    def test_speed_raises_cpu_moe_batch_only_with_ram_headroom(self):
        # Memory-saving CPU-MoE default is a small physical batch; an explicit
        # speed target may raise it when RAM has real headroom.
        no_mtp = model(mtp=False)
        tight = plan_runtime(no_mtp, HardwareProfile("RTX 4060", 8188, 7600, 10000, 16))
        self.assertEqual(tight.ubatch_size, 64)
        fast = plan_runtime(
            no_mtp, HardwareProfile("RTX 4060", 8188, 7600, 10000, 16), mode="speed"
        )
        self.assertEqual((fast.batch_size, fast.ubatch_size), (512, 128))

    def test_speed_never_lowers_an_already_large_batch(self):
        # A fully-resident plan already uses a large batch; speed must keep it.
        big_gpu = HardwareProfile("RTX 4090", 24576, 23000, 50000, 16)
        fast = plan_runtime(model(mtp=False), big_gpu, mode="speed")
        plan = plan_runtime(model(mtp=False), big_gpu)
        self.assertEqual(fast.ubatch_size, plan.ubatch_size)


if __name__ == "__main__":
    unittest.main()
