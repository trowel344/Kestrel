import unittest
import json
import tempfile
import io
from pathlib import Path


class ConverterMathTests(unittest.TestCase):
    def test_source_nvfp4_uses_e2m1_codebook_not_signed_int4(self):
        try:
            import numpy as np
            from kestrel.gguf.converter import dequantize_nvfp4
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        # ModelOpt codes 5, 6, 7 represent 3, 4, 6. A linear INT4 decoder would
        # incorrectly produce 5, 6, 7.
        packed = np.array([[0x65, 0x07]], dtype=np.uint8)
        scales = np.ones((1, 1), dtype=np.float32)
        result = dequantize_nvfp4(packed, scales, 1.0)
        self.assertEqual(result.tolist(), [[3.0, 4.0, 6.0, 0.0]])

    def test_source_to_ggml_nvfp4_round_trip_does_not_double_weights(self):
        try:
            import numpy as np
            from kestrel.gguf.converter import (
                KVALUES,
                dequantize_nvfp4,
                quantize_nvfp4_block,
                ue4m3_to_fp32_vec,
            )
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        # Repeat all 16 source E2M1 codes to make one 64-value GGML block.
        codes = np.tile(np.arange(16, dtype=np.uint8), 4)
        packed = (codes[0::2] | (codes[1::2] << 4)).reshape(1, -1)
        source = dequantize_nvfp4(
            packed,
            np.ones((1, 4), dtype=np.float32),
            1.0,
        )
        encoded = quantize_nvfp4_block(source)

        # Decode the emitted block with llama.cpp's NVFP4 convention: four
        # UE4M3 half-scales followed by four groups of packed E2M1 codes.
        decoded = np.empty((1, 64), dtype=np.float32)
        for group in range(4):
            scale = ue4m3_to_fp32_vec(encoded[:, group])
            values = encoded[:, 4 + group * 8:4 + (group + 1) * 8]
            start = group * 16
            decoded[:, start:start + 8] = scale[:, None] * KVALUES[values & 0x0F]
            decoded[:, start + 8:start + 16] = (
                scale[:, None] * KVALUES[(values >> 4) & 0x0F]
            )

        np.testing.assert_array_equal(decoded, source)

    def test_dense_matrix_payload_stays_source_row_major(self):
        try:
            import numpy as np
            import torch
            from kestrel.gguf.converter import GGML_TYPE_BF16, NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        source = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.bfloat16,
        )
        converter = NVFP4Converter.__new__(NVFP4Converter)
        converter._bf16_hf_key = lambda _name: "source.weight"
        converter._read_torch = lambda _key: source

        payload = converter._load_bf16_tensor_bytes(
            "blk.0.test.weight",
            GGML_TYPE_BF16,
            source.numel() * 2,
        )
        actual = np.frombuffer(payload, dtype=np.uint16)
        expected = source.contiguous().view(torch.uint16).numpy().reshape(-1)

        np.testing.assert_array_equal(actual, expected)

    def test_q4_0_preserves_tiny_double_scaled_expert_values(self):
        try:
            import numpy as np
            from kestrel.gguf.converter import quantize_q4_0
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        source = np.linspace(-0.006, 0.005, 32, dtype=np.float32).reshape(1, 32)
        raw = np.frombuffer(quantize_q4_0(source), dtype=np.uint8).reshape(1, 18)
        scale = raw[:, :2].copy().view(np.float16).astype(np.float32)
        packed = raw[:, 2:]
        decoded = np.empty_like(source)
        decoded[:, :16] = scale * ((packed & 0x0F).astype(np.int8) - 8)
        decoded[:, 16:] = scale * ((packed >> 4).astype(np.int8) - 8)

        correlation = np.corrcoef(source.reshape(-1), decoded.reshape(-1))[0, 1]
        self.assertGreater(correlation, 0.99)
        self.assertGreater(np.count_nonzero(decoded), 28)

    def test_q4_0_decoder_matches_emitted_reference_layout(self):
        try:
            import numpy as np
            from kestrel.gguf.converter import dequantize_q4_0, quantize_q4_0
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        source = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(1, 64)
        encoded = np.frombuffer(quantize_q4_0(source), dtype=np.uint8).reshape(1, -1)
        decoded = dequantize_q4_0(encoded)

        self.assertEqual(decoded.shape, source.shape)
        self.assertGreater(
            np.corrcoef(source.reshape(-1), decoded.reshape(-1))[0, 1],
            0.99,
        )

    def test_q2_k_native_quantizer_shape_and_signal(self):
        try:
            import ctypes
            import numpy as np
            from kestrel.gguf.converter import _load_ggml_base, quantize_q2_k
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"native Q2_K conversion dependency is unavailable: {exc}")

        source = np.linspace(-2.0, 2.0, 512, dtype=np.float32).reshape(2, 256)
        encoded = quantize_q2_k(source)
        self.assertEqual(len(encoded), 2 * 84)

        library = _load_ggml_base()
        decode = library.dequantize_row_q2_K
        decode.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        decode.restype = None
        decoded = np.empty_like(source)
        encoded_buffer = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
        decode(
            ctypes.cast(encoded_buffer, ctypes.c_void_p),
            decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            source.size,
        )
        self.assertGreater(
            np.corrcoef(source.reshape(-1), decoded.reshape(-1))[0, 1],
            0.95,
        )

    def test_iq1_s_native_quantizer_shape_and_signal(self):
        try:
            import ctypes
            import numpy as np
            from kestrel.gguf.converter import _load_ggml_base, quantize_iq1_s
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"native IQ1_S conversion dependency is unavailable: {exc}")

        source = np.sin(np.linspace(-8.0, 8.0, 512, dtype=np.float32)).reshape(2, 256)
        encoded = quantize_iq1_s(source)
        self.assertEqual(len(encoded), 2 * 50)

        library = _load_ggml_base()
        decode = library.dequantize_row_iq1_s
        decode.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        decode.restype = None
        decoded = np.empty_like(source)
        encoded_buffer = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
        decode(
            ctypes.cast(encoded_buffer, ctypes.c_void_p),
            decoded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            source.size,
        )
        self.assertGreater(
            np.corrcoef(source.reshape(-1), decoded.reshape(-1))[0, 1],
            0.95,
        )

    def test_iq1_s_compact_experts_require_direct_primary_conversion(self):
        try:
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        with self.assertRaises(ValueError):
            NVFP4Converter.__new__(NVFP4Converter).__init__(
                compact_expert_type="iq1_s",
                cold_tier="off",
            )
        with self.assertRaises(ValueError):
            NVFP4Converter.__new__(NVFP4Converter).__init__(
                compact_expert_type="iq1_s",
                cold_tier="q1_only",
                experts_only=True,
                imatrix_path="calibration.gguf",
            )
        with self.assertRaisesRegex(ValueError, "--imatrix"):
            NVFP4Converter.__new__(NVFP4Converter).__init__(
                compact_expert_type="iq1_s",
                cold_tier="q1_only",
            )

    def test_q2_edge_layers_require_direct_q1_only_conversion(self):
        try:
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        with self.assertRaises(ValueError):
            NVFP4Converter.__new__(NVFP4Converter).__init__(
                q2_edge_layers=2,
                cold_tier="off",
            )
        with self.assertRaises(ValueError):
            NVFP4Converter.__new__(NVFP4Converter).__init__(
                q2_edge_layers=2,
                cold_tier="q1_only",
                q4_sidecar_source="hot.gguf",
            )

    def test_q4_sidecar_source_requires_q1_only_mode(self):
        try:
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        with self.assertRaises(ValueError):
            NVFP4Converter.__new__(NVFP4Converter).__init__(
                q4_sidecar_source="hot.gguf",
                cold_tier="off",
            )

    def test_experts_only_requires_q1_only_without_mtp(self):
        try:
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        with self.assertRaises(ValueError):
            NVFP4Converter.__new__(NVFP4Converter).__init__(
                experts_only=True,
                cold_tier="off",
            )
        with self.assertRaises(ValueError):
            NVFP4Converter.__new__(NVFP4Converter).__init__(
                experts_only=True,
                cold_tier="q1_only",
                include_mtp=True,
            )

    def test_large_q4_dense_writer_streams_rows_without_changing_layout(self):
        try:
            import numpy as np
            import torch
            from kestrel.gguf.converter import NVFP4Converter, quantize_q4_0
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        source = torch.linspace(-2.0, 2.0, 5 * 64).reshape(5, 64)
        converter = NVFP4Converter.__new__(NVFP4Converter)
        converter._bf16_hf_key = lambda _name: "embed.weight"
        converter._read_torch_slice = lambda _key: source
        output = io.BytesIO()
        expected = quantize_q4_0(source.numpy())

        converter._write_q4_dense(
            output, "token_embd.weight", len(expected), row_chunk=2
        )

        self.assertEqual(output.getvalue(), expected)

    def test_reserved_vocab_ids_are_unused_not_empty_tokens(self):
        try:
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tokenizer.json").write_text(json.dumps({
                "model": {"vocab": {"hello": 0}, "merges": []},
                "added_tokens": [
                    {"id": 1, "content": "<|im_end|>", "special": True},
                ],
            }))
            (root / "tokenizer_config.json").write_text(json.dumps({
                "eos_token": "<|im_end|>",
                "pad_token": "<|im_end|>",
                "chat_template": "{{ messages }}",
            }))
            converter = NVFP4Converter.__new__(NVFP4Converter)
            converter.model_dir = str(root)
            converter.vocab = 4

            tokens, types, _, eos, bos, pad, template = converter._load_tokenizer()

            self.assertEqual(tokens, ["hello", "<|im_end|>", "[PAD2]", "[PAD3]"])
            self.assertEqual(types, [1, 3, 5, 5])
            self.assertEqual((eos, bos, pad), (1, None, 1))
            self.assertEqual(template, "{{ messages }}")

    def test_mtp_is_appended_after_all_target_blocks(self):
        try:
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        converter = NVFP4Converter.__new__(NVFP4Converter)
        converter.tcfg = {
            "rope_parameters": {},
            "full_attention_interval": 2,
            "linear_value_head_dim": 8,
            "linear_num_value_heads": 4,
        }
        converter.n_layer = 2
        converter.mtp_layers = 1
        converter.layer_types = ["linear_attention", "full_attention"]
        converter.hidden = 64
        converter.n_ff = 32
        converter.head_dim = 16
        converter.n_head = 4
        converter.n_kv = 2
        converter.n_exp = 4
        converter.n_used = 2
        converter.vocab = 128
        converter.shared_ff = 32
        converter._load_tokenizer = lambda: None
        converter._read_bf16_info = lambda key: (
            ([4, 8], 64) if key.endswith("conv1d.weight") else ([1], 2)
        )

        converter._init_gguf(None)

        metadata = dict(converter._kvs)
        names = {item[0]: item[3] for item in converter._tensor_infos}
        self.assertEqual(metadata["qwen35moe.block_count"][1], 3)
        self.assertEqual(metadata["qwen35moe.nextn_predict_layers"][1], 1)
        self.assertEqual(names["output_norm.weight"], 0)
        self.assertEqual(names["blk.1.attn_norm.weight"], 0)
        self.assertEqual(names["blk.0.ssm_conv1d.weight"], 0)
        self.assertEqual(names["blk.1.ffn_gate_up_exps.weight"], 2)
        self.assertEqual(names["blk.2.ffn_gate_up_exps.weight"], 1)

    def test_linear_attention_alpha_beta_mapping_matches_llama_cpp(self):
        try:
            from kestrel.gguf.converter import LAYER_MAP_LINEAR
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        self.assertEqual(
            LAYER_MAP_LINEAR["linear_attn.in_proj_a.weight"],
            "blk.{i}.ssm_alpha.weight",
        )
        self.assertEqual(
            LAYER_MAP_LINEAR["linear_attn.in_proj_b.weight"],
            "blk.{i}.ssm_beta.weight",
        )

    def test_linear_attention_v_heads_are_reordered_to_tiled_layout(self):
        try:
            import torch
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        converter = NVFP4Converter.__new__(NVFP4Converter)
        converter.linear_n_k = 2
        converter.linear_n_v = 4
        converter.linear_k_dim = 1
        converter.linear_v_dim = 1

        gate = torch.tensor([[0], [1], [2], [3]])
        actual_gate = converter._transform_linear_attention(
            gate, "model.layers.0.linear_attn.in_proj_z.weight"
        )

        self.assertEqual(actual_gate.flatten().tolist(), [0, 2, 1, 3])

    def test_qwen35_runtime_tensor_transforms_match_llama_cpp(self):
        try:
            import torch
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        converter = NVFP4Converter.__new__(NVFP4Converter)
        converter.linear_n_k = 0
        converter.linear_n_v = 0

        norm = converter._transform_source_tensor(
            torch.tensor([0.0, 0.25]), "model.layers.0.input_layernorm.weight"
        )
        ssm_a = converter._transform_source_tensor(
            torch.tensor([0.0, 1.0]), "model.layers.0.linear_attn.A_log"
        )
        gated_norm = converter._transform_source_tensor(
            torch.tensor([0.0]), "model.layers.0.linear_attn.norm.weight"
        )

        self.assertTrue(torch.equal(norm, torch.tensor([1.0, 1.25])))
        self.assertTrue(torch.allclose(ssm_a, -torch.exp(torch.tensor([0.0, 1.0]))))
        self.assertTrue(torch.equal(gated_norm, torch.tensor([0.0])))

    def test_resolve_pruning_defaults_to_first_kept_experts(self):
        try:
            from kestrel.gguf.converter import _resolve_pruning
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        kept, emitted = _resolve_pruning(256, 8, None)
        self.assertIsNone(kept)
        self.assertEqual(emitted, 256)

        kept, emitted = _resolve_pruning(256, 8, 128)
        self.assertEqual(emitted, 128)
        self.assertEqual(kept, list(range(128)))

    def test_resolve_pruning_rejects_bad_counts(self):
        try:
            from kestrel.gguf.converter import _resolve_pruning
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        with self.assertRaises(ValueError):
            _resolve_pruning(256, 8, 4)  # below experts-per-token
        with self.assertRaises(ValueError):
            _resolve_pruning(256, 8, 256)  # not actually pruning
        with self.assertRaises(ValueError):
            _resolve_pruning(256, 8, 300)  # above the model count

    def test_select_kept_experts_uses_importance_and_ties(self):
        try:
            from kestrel.gguf.converter import _select_kept_experts
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        importance = [0.1, 0.9, 0.5, 0.7, 0.2]
        kept = _select_kept_experts(5, 2, importance)
        self.assertEqual(kept, [1, 3])  # top values, sorted ascending

        ties = [0.0, 0.0, 1.0, 1.0]
        kept = _select_kept_experts(4, 2, ties)
        self.assertEqual(kept, [2, 3])  # tie broken by index

    def test_select_kept_experts_rejects_wrong_length(self):
        try:
            from kestrel.gguf.converter import _select_kept_experts
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        with self.assertRaises(ValueError):
            _select_kept_experts(4, 2, [0.5, 0.5])

    def test_pruned_catalog_uses_emitted_expert_dim(self):
        try:
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        converter = NVFP4Converter.__new__(NVFP4Converter)
        converter.tcfg = {
            "rope_parameters": {},
            "full_attention_interval": 2,
            "linear_value_head_dim": 8,
            "linear_num_value_heads": 4,
        }
        converter.n_layer = 2
        converter.mtp_layers = 0
        converter.layer_types = ["linear_attention", "full_attention"]
        converter.hidden = 64
        converter.n_ff = 32
        converter.head_dim = 16
        converter.n_head = 4
        converter.n_kv = 2
        converter.n_exp = 4
        converter.n_used = 2
        converter._kept = [0, 1]
        converter._emitted_exp = 2
        converter.vocab = 128
        converter.shared_ff = 32
        converter._load_tokenizer = lambda: None
        converter._read_bf16_info = lambda key: (
            ([4, 8], 64) if key.endswith("conv1d.weight") else ([1], 2)
        )

        converter._init_gguf(None)

        dims = {item[0]: item[2] for item in converter._tensor_infos}
        for name, shape in dims.items():
            if ".ffn_gate_up_exps" in name or ".ffn_down_exps" in name:
                self.assertEqual(shape[2], 2, name)
        metadata = dict(converter._kvs)
        self.assertEqual(metadata["qwen35moe.expert_count"][1], 2)

    def test_pruned_catalog_fails_loud_on_dim_mismatch(self):
        try:
            from kestrel.gguf.converter import NVFP4Converter
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        converter = NVFP4Converter.__new__(NVFP4Converter)
        converter.tcfg = {
            "rope_parameters": {},
            "full_attention_interval": 2,
            "linear_value_head_dim": 8,
            "linear_num_value_heads": 4,
        }
        converter.n_layer = 1
        converter.mtp_layers = 0
        converter.layer_types = ["full_attention"]
        converter.hidden = 64
        converter.n_ff = 32
        converter.head_dim = 16
        converter.n_head = 4
        converter.n_kv = 2
        converter.n_exp = 4
        converter.n_used = 2
        converter._kept = [0, 1]
        converter._emitted_exp = 2
        converter.vocab = 128
        converter.shared_ff = 32
        converter.model_dir = "/tmp/opencode/prune-mismatch-model"
        converter._init_gguf = lambda _f: setattr(
            converter,
            "_tensor_infos",
            [("blk.0.ffn_gate_up_exps.weight", 3, [64, 64, 4], 3, 100)],
        )
        converter.experts_only = False

        with self.assertRaises(ValueError):
            converter.convert("/tmp/opencode/prune-mismatch.gguf")


if __name__ == "__main__":
    unittest.main()
