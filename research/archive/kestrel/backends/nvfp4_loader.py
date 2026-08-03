import os
import json
import torch
from safetensors import safe_open


NVFP4_VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    codes = torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1).long()
    values = torch.tensor(NVFP4_VALUES, dtype=torch.int8, device=packed.device)
    return values[codes]


def dequantize_nvfp4_weight(
    packed: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor, group_size: int = 16
) -> torch.Tensor:
    out_dim, packed_in = packed.shape
    in_dim = packed_in * 2
    int4 = unpack_int4(packed)
    s = scale.to(torch.float32).reshape(out_dim, -1, 1)
    groups = int4.reshape(out_dim, -1, group_size)
    return (groups * s).reshape(out_dim, in_dim) * scale_2.to(torch.float32).reshape(-1, 1)


class NVFP4Loader:
    def __init__(self, model_dir: str | None = None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.expanduser("~/.cache/huggingface/hub"),
                "models--nvidia--Qwen3.6-35B-A3B-NVFP4", "snapshots",
                "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
            )
        self.snap_dir = model_dir
        self.available = os.path.exists(os.path.join(self.snap_dir, "model.safetensors.index.json"))
        self._sf: dict[str, safe_open] = {}
        self._weight_map: dict[str, str] | None = None
        self._layer_expert_shard: dict[int, str] | None = None

    @property
    def weight_map(self) -> dict[str, str]:
        if self._weight_map is None:
            with open(os.path.join(self.snap_dir, "model.safetensors.index.json")) as f:
                self._weight_map = json.load(f)["weight_map"]
        return self._weight_map

    def _open(self, shard: str, device: str = "cpu") -> safe_open:
        key = f"{shard}:{device}"
        if key not in self._sf:
            self._sf[key] = safe_open(os.path.join(self.snap_dir, shard), framework="pt", device=device)
        return self._sf[key]

    def _find_shard(self, key: str) -> str | None:
        return self.weight_map.get(key)

    def load_tensor(self, key: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor | None:
        alt = key.removeprefix("model.language_model.")
        found_key = None
        for candidate in [key, alt]:
            if candidate in self.weight_map:
                found_key = candidate
                break
        if found_key is None:
            return None
        sf = self._open(self.weight_map[found_key])
        if found_key not in sf.keys():
            return None
        t = sf.get_tensor(found_key)
        base = found_key.rsplit(".weight", 1)[0] if found_key.endswith(".weight") else found_key
        if t.dtype == torch.uint8 and len(t.shape) == 2:
            sk = f"{base}.weight_scale"
            s2k = f"{base}.weight_scale_2"
            if sk in sf.keys() and s2k in sf.keys():
                return dequantize_nvfp4_weight(t, sf.get_tensor(sk), sf.get_tensor(s2k)).to(dtype=dtype)
        if t.dtype == torch.float8_e4m3fn:
            return t.to(dtype)
        if t.is_floating_point():
            return t.to(dtype=dtype)
        return t.to(dtype=dtype)

    def get_layer_expert_shard(self, layer_idx: int) -> str | None:
        if self._layer_expert_shard is None:
            self._layer_expert_shard = {}
            for key, shard in self.weight_map.items():
                if "mlp.experts." in key and key.endswith(".gate_proj.weight"):
                    layer = int(key.split("layers.")[1].split(".")[0])
                    if layer not in self._layer_expert_shard:
                        self._layer_expert_shard[layer] = shard
        return self._layer_expert_shard.get(layer_idx)

    def load_experts(
        self,
        layer_idx: int,
        expert_indices: list[int],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
        gate_ups = []
        downs = []
        shard_cache = {}
        for eid in expert_indices:
            ep = f"{prefix}.{eid}"
            gp_key = f"{ep}.gate_proj.weight"
            shard_name = self.weight_map.get(gp_key)
            if shard_name is None:
                raise KeyError(f"Missing expert tensor: {gp_key}")
            if shard_name not in shard_cache:
                shard_cache[shard_name] = self._open(shard_name, device=device)
            sf = shard_cache[shard_name]
            gp = dequantize_nvfp4_weight(
                sf.get_tensor(gp_key),
                sf.get_tensor(f"{ep}.gate_proj.weight_scale"),
                sf.get_tensor(f"{ep}.gate_proj.weight_scale_2"),
            ).to(dtype=dtype)
            up = dequantize_nvfp4_weight(
                sf.get_tensor(f"{ep}.up_proj.weight"),
                sf.get_tensor(f"{ep}.up_proj.weight_scale"),
                sf.get_tensor(f"{ep}.up_proj.weight_scale_2"),
            ).to(dtype=dtype)
            gate_ups.append(torch.cat([gp, up], dim=0))
            dn = dequantize_nvfp4_weight(
                sf.get_tensor(f"{ep}.down_proj.weight"),
                sf.get_tensor(f"{ep}.down_proj.weight_scale"),
                sf.get_tensor(f"{ep}.down_proj.weight_scale_2"),
            ).to(dtype=dtype)
            downs.append(dn)
        gu = torch.stack(gate_ups) if gate_ups else torch.empty(0, dtype=dtype, device=device)
        dn = torch.stack(downs) if downs else torch.empty(0, dtype=dtype, device=device)
        return gu, dn

    def load_experts_gpu(
        self, layer_idx: int, expert_indices: list[int], dtype: torch.dtype = torch.bfloat16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.load_experts(
            layer_idx, expert_indices, dtype=dtype, device="cuda"
        )
