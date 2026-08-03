import unittest

from kestrel.gguf.audit import audit_cold_sidecar_snapshot, audit_snapshot


SOURCE_CONFIG = {
    "text_config": {
        "num_hidden_layers": 2,
        "mtp_num_hidden_layers": 1,
    }
}
SOURCE_TOKENIZER = {
    "bos_token": None,
    "pad_token": "<|endoftext|>",
    "chat_template": "{{ messages }}",
}


class GgufAuditTests(unittest.TestCase):
    def test_accepts_complete_q1_cold_sidecar(self):
        fields = {
            "general.architecture": "qwen35moe",
            "qwen35moe.block_count": 2,
        }
        tensors = [
            (f"blk.{layer}.ffn_{projection}_exps_lp.weight", 41)
            for layer in range(2)
            for projection in ("gate_up", "down")
        ]
        findings = audit_cold_sidecar_snapshot(fields, tensors)
        self.assertFalse([item for item in findings if item.severity == "error"])

    def test_cold_sidecar_rejects_missing_wrong_type_and_dense_tensor(self):
        fields = {
            "general.architecture": "qwen35moe",
            "qwen35moe.block_count": 2,
        }
        tensors = [
            ("blk.0.ffn_gate_up_exps_lp.weight", 41),
            ("blk.0.ffn_down_exps_lp.weight", 2),
            ("token_embd.weight", 2),
        ]
        codes = {
            item.code for item in audit_cold_sidecar_snapshot(fields, tensors)
        }
        self.assertIn("SIDECAR_TYPE", codes)
        self.assertIn("SIDECAR_EXTRA_TENSOR", codes)
        self.assertIn("SIDECAR_EXPERTS_MISSING", codes)

    def test_detects_old_conversion_failure_pattern(self):
        fields = {
            "general.architecture": "qwen35moe",
            "qwen35moe.block_count": 2,
            "qwen35moe.nextn_predict_layers": 1,
            "tokenizer.ggml.tokens": ["hello", ""],
            "tokenizer.ggml.bos_token_id": 123,
        }
        tensors = [
            ("blk.0.attn_norm.weight", 0),
            ("blk.1.attn_norm.weight", 0),
            ("blk.1.nextn.eh_proj.weight", 0),
        ]
        findings = audit_snapshot(
            fields,
            tensors,
            source_config=SOURCE_CONFIG,
            source_tokenizer_config=SOURCE_TOKENIZER,
        )
        codes = {item.code for item in findings}
        self.assertIn("MTP_REPLACES_TARGET", codes)
        self.assertIn("BLOCK_COUNT_MISMATCH", codes)
        self.assertIn("TOKEN_TYPES_MISSING", codes)
        self.assertIn("PRETOKENIZER_MISSING", codes)
        self.assertIn("EMPTY_NORMAL_TOKENS", codes)
        self.assertIn("CHAT_TEMPLATE_MISSING", codes)
        self.assertIn("FABRICATED_BOS", codes)
        self.assertIn("PAD_TOKEN_MISSING", codes)

    def test_accepts_target_blocks_with_appended_mtp(self):
        fields = {
            "general.architecture": "qwen35moe",
            "qwen35moe.block_count": 3,
            "qwen35moe.nextn_predict_layers": 1,
            "tokenizer.ggml.tokens": ["hello", "[PAD1]"],
            "tokenizer.ggml.token_type": [1, 5],
            "tokenizer.ggml.pre": "qwen35",
            "tokenizer.chat_template": "{{ messages }}",
            "tokenizer.ggml.padding_token_id": 0,
        }
        tensors = [
            ("blk.0.attn_norm.weight", 30),
            ("blk.1.attn_norm.weight", 30),
            ("blk.2.attn_norm.weight", 30),
            ("blk.2.nextn.eh_proj.weight", 30),
        ]
        findings = audit_snapshot(
            fields,
            tensors,
            source_config=SOURCE_CONFIG,
            source_tokenizer_config=SOURCE_TOKENIZER,
        )
        self.assertFalse([item for item in findings if item.severity == "error"])


if __name__ == "__main__":
    unittest.main()
