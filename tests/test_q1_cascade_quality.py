import unittest


class Q1CascadeQualityTests(unittest.TestCase):
    def test_q1_cosine_matches_decoded_reference(self):
        try:
            import numpy as np
            from kestrel.analysis.q1_cascade_quality import q1_0_cosine
            from kestrel.gguf.converter import quantize_q1_0
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        first = np.linspace(-2.0, 3.0, 256, dtype=np.float32).reshape(2, 128)
        second = first.copy()
        second[:, ::7] *= -1.0
        encoded_first = quantize_q1_0(first)
        encoded_second = quantize_q1_0(second)

        def decode(raw):
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
            scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
            signs = np.unpackbits(blocks[:, 2:], axis=1, bitorder="little")
            return ((signs.astype(np.float32) * 2.0 - 1.0) * scales).reshape(-1)

        a = decode(encoded_first)
        b = decode(encoded_second)
        expected = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        self.assertAlmostEqual(q1_0_cosine(encoded_first, encoded_second), expected, places=6)

    def test_q1_cosine_rejects_invalid_lengths(self):
        try:
            from kestrel.analysis.q1_cascade_quality import q1_0_cosine
        except ImportError as exc:
            self.skipTest(f"conversion dependencies are not installed: {exc}")

        with self.assertRaises(ValueError):
            q1_0_cosine(b"x", b"x")


if __name__ == "__main__":
    unittest.main()
