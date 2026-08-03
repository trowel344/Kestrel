import torch
import triton
import triton.language as tl

@triton.jit
def _nvfp4_dequant_kernel(
    packed_ptr, scale_ptr, scale2_ptr, out_ptr,
    out_dim, in_dim, group_size,
    packed_row_stride, scale_row_stride, out_row_stride,
    BLOCK_IN: tl.constexpr, BLOCK_GROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid

    offs_g = tl.arange(0, BLOCK_GROUPS)
    num_groups = in_dim // group_size

    s = tl.load(scale_ptr + row * scale_row_stride + offs_g, mask=offs_g < num_groups, other=0.0).to(tl.float32)

    offs_in = tl.arange(0, BLOCK_IN)
    packed_off = row * packed_row_stride + offs_in // 2
    packed = tl.load(packed_ptr + packed_off, mask=offs_in < in_dim // 2, other=0)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    vals = tl.where(offs_in % 2 == 0, low, high).to(tl.int8)
    vals = tl.where(vals >= 8, vals - 16, vals).to(tl.float32)

    g_idx = offs_in // group_size
    vals = vals * tl.load(s + g_idx, mask=g_idx < num_groups, other=1.0)

    s2 = tl.load(scale2_ptr).to(tl.float32)
    vals = vals * s2

    tl.store(out_ptr + row * out_row_stride + offs_in, vals.to(tl.bfloat16), mask=offs_in < in_dim)


def dequantize_nvfp4_triton(
    packed: torch.Tensor,
    scale: torch.Tensor,
    scale_2: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    out_dim, packed_in = packed.shape
    in_dim = packed_in * 2
    out = torch.empty(out_dim, in_dim, dtype=torch.bfloat16, device=packed.device)

    BLOCK_IN = 4096
    BLOCK_GROUPS = BLOCK_IN // group_size
    grid = (out_dim,)
    _nvfp4_dequant_kernel[grid](
        packed, scale, scale_2, out,
        out_dim, in_dim, group_size,
        packed.stride(0), scale.stride(0), out.stride(0),
        BLOCK_IN, BLOCK_GROUPS,
    )
    return out
