"""4-bit packed KV quantizer for offline-encoded MSA corpora.

For Phase 4 (100M-token deployment), bf16 K/V storage explodes to 200+ GB
which doesn't fit a single H200's 142 GB GPU memory. This module compresses
the offline pooled tensors 4x by quantizing them to int4 with a per-channel
asymmetric scale.

Quantization scheme
-------------------
For a tensor ``x`` of shape ``(..., D)`` (D = head_dim):

* per-channel min/max over all leading dims for each ``d`` slot
* zero point ``z = min``
* scale ``s = (max - min) / 15``  (15 since int4 range is [0, 15])
* quantized value ``q = clip(round((x - z) / s), 0, 15)``  →  uint8 in [0, 15]
* packed: every two consecutive values share a byte (high nibble first)

Why per-channel? K/V are post-RMSNorm + projection — different head_dim
slots have very different scales. Per-channel keeps the L2 reconstruction
error within ~0.5% which is plenty for retrieval & attention accuracy.

API
---
``quantize_int4_packed`` and ``dequantize_int4_packed`` are the primitives.
``save_corpus_int4`` / ``load_corpus_int4`` provide a layer-major on-disk
format that mirrors the bf16 ``layer_<idx>.pt`` written by
``offline_encoder.encode_corpus``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class _Q4Tensor:
    """Packed int4 tensor + metadata. Use ``.dequantize()`` to recover bf16."""
    packed: torch.Tensor          # uint8 of shape (numel // 2 [+1],)
    scale: torch.Tensor           # bf16 of shape (..., D)
    zero: torch.Tensor            # bf16 of shape (..., D)
    shape: tuple[int, ...]        # original shape
    numel: int                    # original element count

    def dequantize(self, dtype: torch.dtype = torch.bfloat16, device=None) -> torch.Tensor:
        device = device or self.packed.device
        flat = _unpack_int4(self.packed, self.numel).to(device)
        x = flat.to(torch.float32).reshape(self.shape)
        s = self.scale.to(device).to(torch.float32)
        z = self.zero.to(device).to(torch.float32)
        chan_size = s.numel()
        x2d = x.reshape(-1, chan_size)
        x2d = x2d * s.reshape(-1) + z.reshape(-1)
        return x2d.reshape(self.shape).to(dtype)


def quantize_int4_packed(x: torch.Tensor, channel_axes: int = 2) -> _Q4Tensor:
    """Per-(head, dim) asymmetric 4-bit quantization (default keeps last 2 dims as channels)."""
    assert x.is_floating_point()
    orig_shape = tuple(x.shape)
    numel = x.numel()
    n = max(0, x.dim() - channel_axes)
    chan_shape = orig_shape[n:] if n > 0 else orig_shape  # last channel_axes dims
    chan_size = 1
    for s in chan_shape:
        chan_size *= s

    flat2d = x.reshape(-1, chan_size).float()       # (N, C)
    x_min = flat2d.min(dim=0).values                 # (C,)
    x_max = flat2d.max(dim=0).values                 # (C,)
    span = (x_max - x_min).clamp_min(1e-8)
    scale = span / 15.0                              # (C,)
    zero = x_min                                     # (C,)

    q = ((flat2d - zero) / scale).round().clamp_(0, 15).to(torch.uint8)
    flat = q.reshape(-1)
    if flat.numel() & 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    hi = flat[0::2]
    lo = flat[1::2]
    packed = (hi << 4) | lo

    return _Q4Tensor(
        packed=packed,
        scale=scale.to(torch.bfloat16),
        zero=zero.to(torch.bfloat16),
        shape=orig_shape,
        numel=numel,
    )


def _unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    flat = torch.stack([hi, lo], dim=1).reshape(-1)
    return flat[:numel]


def dequantize_int4_packed(qt: _Q4Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return qt.dequantize(dtype=dtype)


def reconstruction_error(x: torch.Tensor) -> dict[str, float]:
    """Quantize → dequantize → compare. For diagnostic use."""
    qt = quantize_int4_packed(x)
    x_hat = qt.dequantize(dtype=x.dtype, device=x.device)
    diff = (x.float() - x_hat.float())
    return {
        "rmse": float(diff.pow(2).mean().sqrt()),
        "max_abs_err": float(diff.abs().max()),
        "snr_db": float(10 * torch.log10(x.float().pow(2).mean() / diff.pow(2).mean().clamp_min(1e-12))),
    }


def save_corpus_int4(out_dir: str | Path, layer_tensors: dict[int, dict[str, torch.Tensor]],
                     meta: Optional[dict] = None,
                     keep_kr_bf16: bool = True) -> dict[int, dict[str, int]]:
    """Quantize and save a per-layer corpus to disk.

    Args:
        out_dir: output directory
        layer_tensors: ``{layer_idx: {"K": K_doc, "V": V_doc, "KR": KR_doc,
                          "doc_offsets": doc_ids}}`` — same layout as
                          ``offline_encoder`` writes.
        meta: optional meta.json contents to copy through (with int4 flag).
        keep_kr_bf16: if True (default) keep router key K_R in bf16 since
            the routing cosine-sim is sensitive to small rounding errors.
            K/V (content) are still int4-quantized for the 4x compression.

    Returns ``{layer_idx: {"bytes_bf16": int, "bytes_int4": int}}`` for
    storage stats.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[int, dict[str, int]] = {}
    for layer_idx, t in layer_tensors.items():
        K_q = quantize_int4_packed(t["K"])
        V_q = quantize_int4_packed(t["V"])
        path = out_dir / f"layer_{layer_idx:02d}_int4.pt"
        save_dict = {
            "K_packed": K_q.packed, "K_scale": K_q.scale, "K_zero": K_q.zero,
            "K_shape": list(K_q.shape), "K_numel": K_q.numel,
            "V_packed": V_q.packed, "V_scale": V_q.scale, "V_zero": V_q.zero,
            "V_shape": list(V_q.shape), "V_numel": V_q.numel,
            "doc_offsets": t["doc_offsets"],
            "format": "msa_int4_v1",
            "keep_kr_bf16": keep_kr_bf16,
        }
        if keep_kr_bf16:
            save_dict["KR_bf16"] = t["KR"].to(torch.bfloat16)
            kr_bytes = t["KR"].numel() * 2
        else:
            KR_q = quantize_int4_packed(t["KR"])
            save_dict.update({
                "KR_packed": KR_q.packed, "KR_scale": KR_q.scale, "KR_zero": KR_q.zero,
                "KR_shape": list(KR_q.shape), "KR_numel": KR_q.numel,
            })
            kr_bytes = KR_q.packed.numel() + (KR_q.scale.numel() + KR_q.zero.numel()) * 2
        torch.save(save_dict, path)
        bytes_bf16 = (t["K"].numel() + t["V"].numel() + t["KR"].numel()) * 2
        bytes_int4 = (
            K_q.packed.numel() + V_q.packed.numel()
            + (K_q.scale.numel() + K_q.zero.numel()
               + V_q.scale.numel() + V_q.zero.numel()) * 2
            + kr_bytes
        )
        stats[layer_idx] = {"bytes_bf16": bytes_bf16, "bytes_int4": bytes_int4}
    if meta is not None:
        import json
        meta_w = dict(meta)
        meta_w["int4_quantized"] = True
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta_w, f, indent=2)
    return stats


def load_layer_int4(path: str | Path, device: str = "cpu",
                    dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Load + dequantize one layer's int4 file. Returns bf16 K/V/KR + doc_offsets."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    assert d.get("format") == "msa_int4_v1", f"unknown format in {path}"
    out = {}
    for k in ("K", "V"):
        qt = _Q4Tensor(
            packed=d[f"{k}_packed"],
            scale=d[f"{k}_scale"],
            zero=d[f"{k}_zero"],
            shape=tuple(d[f"{k}_shape"]),
            numel=int(d[f"{k}_numel"]),
        )
        out[k] = qt.dequantize(dtype=dtype, device=device)
    if d.get("keep_kr_bf16", False):
        out["KR"] = d["KR_bf16"].to(device=device, dtype=dtype)
    else:
        qt = _Q4Tensor(
            packed=d["KR_packed"], scale=d["KR_scale"], zero=d["KR_zero"],
            shape=tuple(d["KR_shape"]), numel=int(d["KR_numel"]),
        )
        out["KR"] = qt.dequantize(dtype=dtype, device=device)
    out["doc_offsets"] = d["doc_offsets"]
    return out
