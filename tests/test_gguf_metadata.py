import struct
import tempfile
import unittest
from pathlib import Path

from kestrel.gguf.metadata import GGUFMetadataError, read_planner_metadata


def gguf_string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def kv_string(key: str, value: str) -> bytes:
    return gguf_string(key) + struct.pack("<I", 8) + gguf_string(value)


def kv_uint32(key: str, value: int) -> bytes:
    return gguf_string(key) + struct.pack("<II", 4, value)


class GGUFMetadataTests(unittest.TestCase):
    def test_reads_planner_fields_without_entering_tokenizer_payload(self):
        fields = [
            kv_string("general.architecture", "qwen35moe"),
            kv_uint32("qwen35moe.vision.block_count", 24),
            kv_uint32("qwen35moe.vision.embedding_length", 1024),
            kv_uint32("qwen35moe.block_count", 48),
            kv_uint32("qwen35moe.expert_count", 256),
            kv_uint32("qwen35moe.expert_used_count", 8),
            kv_uint32("qwen35moe.embedding_length", 3072),
            kv_uint32("qwen35moe.expert_feed_forward_length", 1024),
            kv_uint32("qwen35moe.nextn_predict_layers", 1),
            # Deliberately truncated: the reader must stop before a giant
            # tokenizer array once planner metadata is complete.
            gguf_string("tokenizer.ggml.tokens") + struct.pack("<I", 9),
        ]
        payload = b"GGUF" + struct.pack("<IQQ", 3, 999, len(fields)) + b"".join(fields)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            path.write_bytes(payload)
            result = read_planner_metadata(path)

        self.assertEqual(result["architecture"], "qwen35moe")
        self.assertEqual(result["n_layer"], 48)
        self.assertEqual(result["n_exp"], 256)
        self.assertEqual(result["mtp_layers"], 1)

    def test_rejects_non_gguf_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gguf"
            path.write_bytes(b"nope")
            with self.assertRaises(GGUFMetadataError):
                read_planner_metadata(path)

    def test_dense_model_defaults_expert_counts_to_zero(self):
        fields = [
            kv_string("general.architecture", "qwen2"),
            kv_uint32("qwen2.block_count", 28),
            kv_uint32("qwen2.embedding_length", 3584),
            kv_uint32("qwen2.feed_forward_length", 18944),
            gguf_string("tokenizer.ggml.tokens") + struct.pack("<I", 9),
        ]
        payload = b"GGUF" + struct.pack("<IQQ", 3, 100, len(fields)) + b"".join(fields)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.gguf"
            path.write_bytes(payload)
            result = read_planner_metadata(path)
        self.assertEqual(result["n_exp"], 0)
        self.assertEqual(result["n_used"], 0)
        self.assertEqual(result["n_ff"], 18944)


if __name__ == "__main__":
    unittest.main()
