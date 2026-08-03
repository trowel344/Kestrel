import json
import unittest


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class KimiProviderTests(unittest.TestCase):
    def test_requires_explicit_secret(self):
        from kestrel.providers.kimi import KimiClient, KimiError

        client = KimiClient(api_key="", opener=lambda *_args, **_kwargs: None)
        client.api_key = None
        with self.assertRaisesRegex(KimiError, "KIMI_API_KEY"):
            client.complete([{"role": "user", "content": "hello"}])

    def test_preserves_reasoning_message_for_followup_turns(self):
        from kestrel.providers.kimi import KimiClient

        captured = {}

        def opener(request, **_kwargs):
            captured["request"] = request
            return _Response({
                "choices": [{"message": {
                    "role": "assistant",
                    "reasoning_content": "private working state",
                    "content": "answer",
                }}],
                "usage": {"completion_tokens": 2},
            })

        client = KimiClient(api_key="secret", opener=opener)
        result = client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(result.content, "answer")
        self.assertEqual(result.reasoning_content, "private working state")
        self.assertEqual(result.raw_message["role"], "assistant")
        payload = json.loads(captured["request"].data)
        self.assertEqual(payload["model"], "kimi-k3")
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("temperature", payload)


if __name__ == "__main__":
    unittest.main()
