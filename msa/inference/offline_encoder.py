"""Offline corpus encoder for MSA-Qwen3.5-9B inference.

For each MSA layer in our trained MSAQwen3_5ForCausalLM, this module:

  1. Tokenises a corpus of documents with the EverMind ``[idx]. doc[idx]``
     wrapper (see :mod:`msa.inference.prompt_template`).
  2. Runs ``MSAQwen3_5Model.encode_docs`` over the corpus in mini-batches.
  3. Concatenates the per-layer pooled ``(K_bar, V_bar, Kr_bar)`` tensors
     across all batches.
  4. Persists them to disk under ``<out_dir>/layer_<i>.pt`` as a single
     dict ``{"K": ..., "V": ..., "KR": ..., "doc_offsets": ...}``.

The router-engine + sparse-generator load these tensors back to do top-k
retrieval at inference time.

This is the stage-1 prefill of the MSA paper §3.2 protocol, adapted to
Qwen3.5's hybrid backbone (only the 8 full-attention layers contribute
to the pooled cache; the 24 GatedDeltaNet layers run in tandem so each
doc gets a fresh recurrent state, but they don't produce any pooled K/V).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from msa.model_msa_qwen3_5 import MSAQwen3_5Config, MSAQwen3_5ForCausalLM
from msa.inference.memory_loader import BenchmarkData


@dataclass
class EncoderConfig:
    """Knobs for the offline encoder."""

    max_doc_len: int = 128
    batch_size: int = 32          # docs per encode call
    docs_per_batch_dim: int = 32  # collapse all docs in one batch into N axis
    dtype: str = "bf16"           # bf16 / fp16 / fp32
    device: str = "cuda"


def _truncate_doc(text: str, tokenizer, max_doc_len: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:max_doc_len]


class _DocDataset(Dataset):
    """Tokenise + pad a list of documents to fixed length."""

    def __init__(self, docs: Sequence[str], tokenizer, max_doc_len: int):
        self.tokenizer = tokenizer
        self.max_doc_len = max_doc_len
        self.docs = docs

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int):
        ids = _truncate_doc(self.docs[idx], self.tokenizer, self.max_doc_len)
        n = len(ids)
        padded = ids + [self.tokenizer.pad_token_id] * (self.max_doc_len - n)
        mask = [1] * n + [0] * (self.max_doc_len - n)
        return {
            "doc_idx": idx,
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def _collate(batch):
    return {
        "doc_idx": torch.tensor([b["doc_idx"] for b in batch], dtype=torch.long),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


def encode_corpus(
    documents: Sequence[str],
    model: MSAQwen3_5ForCausalLM,
    tokenizer,
    cfg: EncoderConfig,
    out_dir: str | Path,
    log_every: int = 16,
    shard_size: int = 100_000,
    merge_shards: bool = True,
) -> dict[int, dict]:
    """Encode an entire corpus and save per-layer pooled KVR to disk.

    Streams to disk in shards of ``shard_size`` documents (default 100k) so
    that peak CPU memory stays bounded — for 100M-token corpora the prior
    "accumulate everything in RAM, then concat" path OOMs around ~150 GB
    of CPU usage. Each shard is dumped as ``layer_<NN>_shard_<MM>.pt``.

    If ``merge_shards`` is True (default), once encoding finishes we walk
    each layer and merge its shards into a single ``layer_<NN>.pt`` file
    (peak memory ~one layer's worth, much smaller than full corpus).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.dtype]

    msa_layer_indices = list(model.config.msa_layer_indices or [])
    msa_layer_indices = sorted(msa_layer_indices)

    ds = _DocDataset(documents, tokenizer, cfg.max_doc_len)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    n_total = len(documents)
    print(
        f"[offline_encoder] encoding {n_total} docs, "
        f"max_doc_len={cfg.max_doc_len}, batch_size={cfg.batch_size}, "
        f"MSA layers={msa_layer_indices}, shard_size={shard_size}"
    )

    def _new_accum():
        return {
            i: {"K": [], "V": [], "KR": [], "doc_offsets": []}
            for i in msa_layer_indices
        }

    accum = _new_accum()
    docs_in_shard = 0
    shard_idx = 0
    docs_total_seen = 0
    shard_files: dict[int, list[str]] = {i: [] for i in msa_layer_indices}
    t0 = time.time()
    for step, batch in enumerate(loader):
        # Reshape (B, L) → (1, N, L) so encode_docs sees one outer batch with
        # N parallel docs. This matches the training layout.
        n = batch["input_ids"].shape[0]
        doc_input_ids = batch["input_ids"].to(device).unsqueeze(0)
        doc_attention_mask = batch["attention_mask"].to(device).unsqueeze(0)

        pooled = model.model.encode_docs(
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
        )

        for layer_idx, (K_bar, V_bar, Kr_bar) in pooled.items():
            # K_bar: (1, N, n_chunks, n_kv_heads, head_dim) — bf16 / fp16
            accum[layer_idx]["K"].append(K_bar.squeeze(0).to(dtype).cpu())
            accum[layer_idx]["V"].append(V_bar.squeeze(0).to(dtype).cpu())
            accum[layer_idx]["KR"].append(Kr_bar.squeeze(0).to(dtype).cpu())

        # Track which docs landed in this batch (global indices)
        for layer_idx in msa_layer_indices:
            accum[layer_idx]["doc_offsets"].append(batch["doc_idx"].cpu())

        docs_in_shard += n
        docs_total_seen += n

        is_last_step = (step + 1 == len(loader))
        if (step + 1) % log_every == 0 or is_last_step:
            elapsed = time.time() - t0
            rate = docs_total_seen / max(elapsed, 1e-6)
            print(
                f"  step {step+1}/{len(loader)} | "
                f"docs={docs_total_seen}/{n_total} | "
                f"{rate:.1f} docs/s | elapsed={elapsed:.1f}s"
            )

        # Flush a shard
        if docs_in_shard >= shard_size or is_last_step:
            for layer_idx in msa_layer_indices:
                K = torch.cat(accum[layer_idx]["K"], dim=0)
                V = torch.cat(accum[layer_idx]["V"], dim=0)
                KR = torch.cat(accum[layer_idx]["KR"], dim=0)
                doc_offsets = torch.cat(accum[layer_idx]["doc_offsets"], dim=0)
                shard_path = out_dir / f"layer_{layer_idx:02d}_shard_{shard_idx:03d}.pt"
                torch.save(
                    {"K": K, "V": V, "KR": KR,
                     "doc_offsets": doc_offsets,
                     "layer_idx": layer_idx,
                     "shard_idx": shard_idx},
                    shard_path,
                )
                shard_files[layer_idx].append(str(shard_path))
            print(f"  [shard {shard_idx}] flushed {docs_in_shard} docs "
                  f"(total {docs_total_seen}/{n_total})")
            accum = _new_accum()
            docs_in_shard = 0
            shard_idx += 1

    print(f"[offline_encoder] encoding done in {time.time() - t0:.1f}s")

    meta_summary: dict[int, dict] = {}
    if merge_shards:
        print(f"[offline_encoder] merging {shard_idx} shards per layer ...")
        for layer_idx in msa_layer_indices:
            K_parts, V_parts, KR_parts, off_parts = [], [], [], []
            for sp in shard_files[layer_idx]:
                d = torch.load(sp, map_location="cpu", weights_only=True)
                K_parts.append(d["K"])
                V_parts.append(d["V"])
                KR_parts.append(d["KR"])
                off_parts.append(d["doc_offsets"])
            K = torch.cat(K_parts, dim=0)
            V = torch.cat(V_parts, dim=0)
            KR = torch.cat(KR_parts, dim=0)
            doc_offsets = torch.cat(off_parts, dim=0)
            del K_parts, V_parts, KR_parts, off_parts
            out_path = out_dir / f"layer_{layer_idx:02d}.pt"
            torch.save(
                {"K": K, "V": V, "KR": KR,
                 "doc_offsets": doc_offsets,
                 "layer_idx": layer_idx},
                out_path,
            )
            meta_summary[layer_idx] = {
                "path": str(out_path),
                "K_shape": list(K.shape),
                "V_shape": list(V.shape),
                "KR_shape": list(KR.shape),
                "size_mb": (K.numel() + V.numel() + KR.numel()) * K.element_size() / 1e6,
            }
            del K, V, KR, doc_offsets
            for sp in shard_files[layer_idx]:
                Path(sp).unlink(missing_ok=True)
    else:
        for layer_idx in msa_layer_indices:
            meta_summary[layer_idx] = {
                "shards": shard_files[layer_idx],
                "n_shards": len(shard_files[layer_idx]),
            }

    summary_path = out_dir / "meta.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "num_documents": n_total,
                "max_doc_len": cfg.max_doc_len,
                "batch_size": cfg.batch_size,
                "dtype": cfg.dtype,
                "msa_layer_indices": msa_layer_indices,
                "shard_size": shard_size,
                "n_shards": shard_idx,
                "merged": merge_shards,
                "layers": meta_summary,
            },
            f,
            indent=2,
        )
    print(f"[offline_encoder] wrote {summary_path}")
    if merge_shards:
        print(f"[offline_encoder] per-layer storage:")
        for layer_idx, m in meta_summary.items():
            print(f"  L{layer_idx:2d}  K={m['K_shape']} V={m['V_shape']} "
                  f"KR={m['KR_shape']}  total={m['size_mb']:.1f} MB")
    return meta_summary
