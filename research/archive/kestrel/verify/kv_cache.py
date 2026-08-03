import torch
from dataclasses import dataclass


@dataclass
class KVCacheEntry:
    keys: torch.Tensor
    values: torch.Tensor
    seq_len: int


class KVCacheManager:
    def __init__(
        self,
        num_full_attn_layers: int = 10,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        max_seq_len: int = 32768,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_layers = num_full_attn_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device)
        self.kv_dim = num_kv_heads * head_dim

        self.caches: list[KVCacheEntry | None] = [None] * num_full_attn_layers
        self.compression_ratio = 6.0
        self.compression_enabled = False
        self.allocated_vram = 0

    def reset(self):
        self.caches = [None] * self.num_layers
        self.allocated_vram = 0

    def _layer_to_cache_idx(self, global_layer_idx: int) -> int:
        return global_layer_idx // 4

    def update(self, global_layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        if not self._is_full_attention(global_layer_idx):
            return
        idx = self._layer_to_cache_idx(global_layer_idx)
        entry = self.caches[idx]
        if entry is None:
            self.caches[idx] = KVCacheEntry(keys=key, values=value, seq_len=key.shape[2])
            if self.compression_enabled:
                compressed = self.kv_dim * key.shape[2] / self.max_seq_len
                self.allocated_vram += compressed * 2
            else:
                self.allocated_vram += key.numel() * key.element_size() * 2
        else:
            entry.keys = torch.cat([entry.keys, key], dim=2)
            entry.values = torch.cat([entry.values, value], dim=2)
            entry.seq_len = entry.keys.shape[2]

    def get(self, global_layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._is_full_attention(global_layer_idx):
            return None
        idx = self._layer_to_cache_idx(global_layer_idx)
        entry = self.caches[idx]
        if entry is None:
            return None
        return entry.keys, entry.values

    def _is_full_attention(self, global_layer_idx: int) -> bool:
        return global_layer_idx % 4 == 3

    @property
    def total_seq_len(self) -> int:
        max_len = 0
        for entry in self.caches:
            if entry is not None:
                max_len = max(max_len, entry.seq_len)
        return max_len

    @property
    def vram_usage_gb(self) -> float:
        return self.allocated_vram / 1024**3

    def summary(self) -> dict:
        filled = sum(1 for e in self.caches if e is not None)
        return {
            "layers_active": f"{filled}/{self.num_layers}",
            "total_seq_len": self.total_seq_len,
            "vram_gb": round(self.vram_usage_gb, 3),
            "compression": f"{self.compression_ratio}x" if self.compression_enabled else "off",
            "kv_dim": self.kv_dim,
        }
