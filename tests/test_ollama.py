import json
import urllib.error

import pytest

from kestrel.providers.ollama import OllamaClient, OllamaError


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, amount=None):
        return self._payload if amount is None else self._payload[:amount]


class _FakeHTTPError(Exception):
    pass


def _success_opener(payload: dict):
    body = json.dumps(payload).encode()

    def opener(request, timeout):
        return _FakeResponse(body)

    return opener


def test_generate_parses_rates():
    payload = {
        "model": "qwen3.6:27b",
        "response": "hello",
        "thinking": "",
        "prompt_eval_count": 10,
        "prompt_eval_duration": 5_000_000_000,  # 5s
        "eval_count": 20,
        "eval_duration": 2_000_000_000,  # 2s
        "total_duration": 7_000_000_000,
    }
    client = OllamaClient(base_url="http://localhost:11434", opener=_success_opener(payload))
    result = client.generate("qwen3.6:27b", "hi", num_predict=8)
    assert result.response == "hello"
    assert result.prompt_tps == pytest.approx(2.0)
    assert result.decode_tps == pytest.approx(10.0)
    assert result.prompt_tokens == 10
    assert result.generated_tokens == 20


def test_http_error_wrapped():
    def opener(request, timeout):
        raise urllib.error.HTTPError("url", 500, "boom", {}, io_error_reader_factory())

    client = OllamaClient(base_url="http://localhost:11434", opener=opener)
    with pytest.raises(OllamaError):
        client.generate("m", "p")


def io_error_reader_factory():
    class Reader:
        def read(self, amt=None):
            return b"server exploded"

        def close(self):
            return None

    return Reader()


def test_connection_error_wrapped():
    def opener(request, timeout):
        raise urllib.error.URLError("no server")

    client = OllamaClient(base_url="http://localhost:11434", opener=opener)
    with pytest.raises(OllamaError, match="Could not reach Ollama"):
        client.generate("m", "p")


def test_invalid_json_wrapped():
    def opener(request, timeout):
        return _FakeResponse(b"not json {")

    client = OllamaClient(base_url="http://localhost:11434", opener=opener)
    with pytest.raises(OllamaError, match="invalid JSON"):
        client.generate("m", "p")


def test_oversized_response_is_rejected_without_unbounded_read():
    response = _FakeResponse(b"x" * 17)
    client = OllamaClient(base_url="http://localhost:11434", opener=lambda *args, **kwargs: response)
    client._MAX_RESPONSE_BYTES = 16

    with pytest.raises(OllamaError, match="exceeded"):
        client.generate("m", "p")


def test_invalid_utf8_is_wrapped():
    client = OllamaClient(
        base_url="http://localhost:11434",
        opener=lambda *args, **kwargs: _FakeResponse(b'"\xff"'),
    )
    with pytest.raises(OllamaError, match="invalid JSON"):
        client.generate("m", "p")


def test_non_object_json_is_wrapped():
    def opener(request, timeout):
        return _FakeResponse(b"[]")

    client = OllamaClient(base_url="http://localhost:11434", opener=opener)
    with pytest.raises(OllamaError, match="unexpected JSON response"):
        client.generate("m", "p")


@pytest.mark.parametrize("value", ["many", True, 1.5, -1, None, float("nan"), float("inf"), 1 << 64])
def test_invalid_metrics_are_wrapped(value):
    client = OllamaClient(
        base_url="http://localhost:11434",
        opener=_success_opener({"response": "ok", "eval_count": value}),
    )
    with pytest.raises(OllamaError, match="invalid generation metrics"):
        client.generate("m", "p")


@pytest.mark.parametrize("field", ["response", "thinking"])
def test_invalid_generation_text_is_wrapped(field):
    client = OllamaClient(
        base_url="http://localhost:11434",
        opener=_success_opener({field: ["not", "text"]}),
    )
    with pytest.raises(OllamaError, match="invalid generation text"):
        client.generate("m", "p")


def test_invalid_or_oversized_error_detail_is_bounded():
    malformed = OllamaClient(base_url="http://localhost:11434", opener=_success_opener({"error": ["bad"]}))
    with pytest.raises(OllamaError, match="invalid error response"):
        malformed.generate("m", "p")

    bounded = OllamaClient(base_url="http://localhost:11434", opener=_success_opener({"error": "x" * 2000}))
    with pytest.raises(OllamaError) as caught:
        bounded.generate("m", "p")
    assert len(str(caught.value)) == 1000


def test_huge_json_integer_is_wrapped(monkeypatch):
    monkeypatch.setattr("sys.get_int_max_str_digits", lambda: 4300, raising=False)
    payload = b'{"eval_count": ' + (b"9" * 5000) + b"}"
    client = OllamaClient(base_url="http://localhost:11434", opener=lambda *_args, **_kwargs: _FakeResponse(payload))
    with pytest.raises(OllamaError, match="invalid JSON"):
        client.generate("m", "p")


def test_error_field_wrapped():
    payload = {"error": "model not found"}
    client = OllamaClient(base_url="http://localhost:11434", opener=_success_opener(payload))
    with pytest.raises(OllamaError, match="model not found"):
        client.generate("m", "p")


def test_base_url_normalization():
    client = OllamaClient(base_url="localhost:11434")
    assert client.base_url == "http://localhost:11434"
    client2 = OllamaClient(base_url="http://localhost:11434")
    assert client2.base_url == "http://localhost:11434"


def test_zero_durations_no_rates():
    payload = {
        "model": "m",
        "response": "x",
        "thinking": "",
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
        "total_duration": 0,
    }
    client = OllamaClient(base_url="http://localhost:11434", opener=_success_opener(payload))
    result = client.generate("m", "p")
    assert result.prompt_tps is None
    assert result.decode_tps is None
