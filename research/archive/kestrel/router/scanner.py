import torch
from ..backends.nvfp4_loader import NVFP4Loader


class RouterScanner:
    def __init__(
        self,
        n_layers: int,
        n_experts: int,
        n_embd: int,
        top_k: int = 8,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_embd = n_embd
        self.top_k = top_k
        self.device = torch.device(device)
        self.dtype = dtype
        self._gate_weights: torch.Tensor | None = None

    def load_from_model_dir(self, model_dir: str) -> None:
        loader = NVFP4Loader(model_dir)
        gates = []
        for i in range(self.n_layers):
            key = f"model.language_model.layers.{i}.mlp.gate.weight"
            t = loader.load_tensor(key, dtype=self.dtype)
            if t is None:
                raise KeyError(f"Router tensor is missing: {key}")
            gates.append(t.to(device=self.device))
        self._gate_weights = torch.stack(gates, dim=0)

    def _ensure_loaded(self):
        if self._gate_weights is None:
            raise RuntimeError("Router weights not loaded. Call load_from_model_dir() first.")

    @torch.inference_mode()
    def scan(self, hidden_states: torch.Tensor) -> list[list[int]]:
        """Approximate all-layer routing from a shared hidden state.

        This is not an exact router-first pass: deeper routers normally consume
        hidden states produced by preceding layers. Callers must treat this as
        a prefetch prediction only.
        """
        self._ensure_loaded()
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.n_embd:
            raise ValueError(
                f"Expected hidden states shaped [batch, {self.n_embd}], "
                f"got {tuple(hidden_states.shape)}"
            )
        hs = hidden_states.to(dtype=self.dtype, device=self.device)
        logits = torch.einsum("bn,lmn->blm", hs, self._gate_weights)
        probs = torch.softmax(logits.float(), dim=-1)
        _, top_indices = torch.topk(probs, self.top_k, dim=-1)
        return top_indices.squeeze(0).tolist()

    @torch.inference_mode()
    def scan_single_layer(self, layer_idx: int, hidden_states: torch.Tensor) -> list[int]:
        self._ensure_loaded()
        w = self._gate_weights[layer_idx]
        hs = hidden_states.to(dtype=self.dtype, device=self.device)
        logits = hs @ w.T
        probs = torch.softmax(logits.float(), dim=-1)
        _, top_indices = torch.topk(probs, self.top_k, dim=-1)
        return top_indices.tolist()
