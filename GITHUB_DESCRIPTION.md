# Kestrel

Hardware-aware local model launcher, converter, and manager built on llama.cpp.
It profiles your machine, plans memory to avoid OOM, converts Qwen3.5-MoE NVFP4
to GGUF, and manages disk / Hugging Face / Ollama models from one CLI.

## Why Kestrel

llama.cpp, but it plans the memory for you. Profiles your GPU/RAM, then only
enables what your binary can actually do.

## Models tested

- Qwen3.5-122B-A10B (MoE, 48 layers, ~72 GiB GGUF) on RTX 4060 Laptop (8 GiB VRAM)
- Qwen3.5-4B GGUF

## Measured numbers

| Metric | Value |
|---|---|
| Qwen3.5-122B decode | 6.33 generated tok/s (8 GiB GPU, IQ1_S profile) |
| GGUF metadata inspect (72.2 GiB) | ~0.05 s |
