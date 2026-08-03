# Contributing to Kestrel

Kestrel is deliberately a small orchestrator around real inference engines.
Changes should improve a measured supported path rather than add an unverified
second runtime.

## Development

```bash
python -m pip install -e .
python scripts/run_test.py
python -m compileall -q kestrel scripts
python -m build
```

Conversion development additionally needs `python -m pip install -e
'.[convert]'` and a compatible llama.cpp `libggml-base.so` for native K-quants.

Pull requests should include the hardware, exact model artifact, complete
command and prompt/decode measurements for performance claims. Quantization or
placement changes also require deterministic accuracy and stability results.
Do not commit model weights, API keys, private prompts or multi-gigabyte logs.
