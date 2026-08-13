import io
import threading
import time

from kestrel.cli import telemetry


def test_metric_value_parses_known_token_rate():
    body = (
        "# HELP llm_tokens_per_second Current token rate\n"
        "# TYPE llm_tokens_per_second gauge\n"
        'llm_tokens_per_second{model="m",gpu="0"} 42.5\n'
        "llm_tokens_total 1234\n"
    )
    assert telemetry._metric_value(body) == 42.5
    assert telemetry._metric_value("# comment only\nsome_other 1.0\n") is None


def test_server_tps_returns_none_when_metrics_unavailable(monkeypatch):
    def unavailable(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(telemetry.urllib.request, "urlopen", unavailable)
    assert telemetry._server_tps("127.0.0.1", 8080) is None


def test_server_tps_parses_metrics_response(monkeypatch):
    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _limit):
            return b'llm_tokens_per_second{gpu="0"} 17.25\n'

    monkeypatch.setattr(telemetry.urllib.request, "urlopen", lambda url, timeout: Resp())
    assert telemetry._server_tps("127.0.0.1", 8080) == 17.25


def test_live_dashboard_writes_line_and_stops():
    out = io.StringIO()
    stop = threading.Event()

    thread = threading.Thread(
        target=telemetry.live_dashboard,
        args=(stop,),
        kwargs={"host": "127.0.0.1", "port": 8080, "interval": 0.01, "out": out},
    )
    thread.start()
    time.sleep(0.05)
    stop.set()
    thread.join(timeout=2)

    assert "Kestrel" in out.getvalue()
