"""Online router for MSA-Qwen3.5-9B inference.

Given a corpus that has already been pushed through
:func:`msa.inference.offline_encoder.encode_corpus`, this module:

  1. Loads the per-layer pooled ``(K_bar, V_bar, Kr_bar)`` tensors from disk.
  2. For each new query, runs the trained MSAQwen3_5 model up to each MSA
     layer to produce ``Q_r`` projections.
  3. Scores ``Q_r`` against the corpus ``Kr_bar`` to pick top-k chunks.
  4. Returns chunk indices + doc indices that the sparse generator will
     attend to.

This is the §3.2 stage-2 protocol of the MSA paper, adapted to Qwen3.5.

For correctness, the router uses **exactly the same** scoring rule as the
model's training-mode forward (``MSAQwen3_5DecoderLayer.query_msa_attention``
in ``model_msa_qwen3_5.py``):

    Qr_n = F.normalize(Qr.float(), dim=-1)
    Kr_n = F.normalize(Kr_bar_q.float(), dim=-1)
    cos_sim = einsum("blhd,bnchd->blhnc", Qr_n, Kr_n)
    sim_chunk = cos_sim.mean(dim=2)        # over router heads
    sim_chunk = sim_chunk.amax(dim=1)      # over query positions
    s_per_doc = sim_chunk.amax(dim=-1)     # over chunks → [B, N]
    top_k_idx = s_per_doc.topk(k, dim=-1)

The only difference vs training is that ``Kr_bar`` here comes from disk
instead of being recomputed from ``doc_input_ids`` every step, and the
top-k is over **all** corpus chunks (not just the in-batch docs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# In-memory representation of an offline-encoded corpus
# -----------------------------------------------------------------------------
@dataclass
class EncodedCorpus:
    """Per-MSA-layer pooled ``(K, V, K_R)`` plus doc-offset metadata.

    Stored on the chosen ``device``. For 100M corpora the full bf16 tensors
    will not fit on a single GPU and we'll page them chunk-wise; this small
    container is what the router-engine works with for the in-memory case
    (short / medium benchmarks).

    Two views are kept in sync so callers can pick whichever they need:
      * Flat:  ``K``/``V``/``KR`` of shape ``(n_chunks_total, n_heads, dim)``
        — for fast top-k routing across the whole corpus.
      * Doc-grouped: ``K_doc``/``V_doc``/``KR_doc`` of shape
        ``(n_docs, n_chunks_per_doc, n_heads, dim)`` — for the
        ``forward_query_msa`` shape contract (B, N, n_chunks, …).

    ``chunk_to_doc``: (n_chunks_total,) — index of the source document in the
        original corpus order.
    ``doc_ids``: (n_docs,) — global doc ids for each row of the doc axis.
    """

    layer_idx: int
    # Flat (per-chunk) view
    K: torch.Tensor
    V: torch.Tensor
    KR: torch.Tensor
    chunk_to_doc: torch.Tensor
    # Doc-grouped view
    K_doc: torch.Tensor
    V_doc: torch.Tensor
    KR_doc: torch.Tensor
    doc_ids: torch.Tensor

    @property
    def n_chunks(self) -> int:
        return self.K.shape[0]

    @property
    def n_docs(self) -> int:
        return self.K_doc.shape[0]

    @property
    def n_chunks_per_doc(self) -> int:
        return self.K_doc.shape[1]

    def to(self, device, dtype=None) -> "EncodedCorpus":
        def _t(x):
            return x.to(device=device, dtype=dtype) if (dtype and x.is_floating_point()) else x.to(device)
        return EncodedCorpus(
            layer_idx=self.layer_idx,
            K=_t(self.K), V=_t(self.V), KR=_t(self.KR),
            chunk_to_doc=self.chunk_to_doc.to(device),
            K_doc=_t(self.K_doc), V_doc=_t(self.V_doc), KR_doc=_t(self.KR_doc),
            doc_ids=self.doc_ids.to(device),
        )

    def insert_doc(
        self,
        K_doc_new: torch.Tensor,
        V_doc_new: torch.Tensor,
        KR_doc_new: torch.Tensor,
        position: int,
        new_doc_id: int = -1,
    ) -> "EncodedCorpus":
        """Return a new EncodedCorpus with one extra doc spliced at ``position``.

        Used by the NIAH harness to inject a synthetic needle doc into a
        pre-encoded haystack without re-encoding the entire corpus. ``K_doc_new``
        et al. should have shape ``(n_chunks_per_doc, n_heads, dim)`` — i.e. the
        same layout as one row of ``self.K_doc`` minus the leading doc axis.
        """
        n_docs = self.n_docs
        if not (0 <= position <= n_docs):
            raise ValueError(f"position {position} out of [0,{n_docs}]")
        if K_doc_new.shape != self.K_doc.shape[1:]:
            raise ValueError(
                f"needle K shape {tuple(K_doc_new.shape)} != "
                f"corpus per-doc K shape {tuple(self.K_doc.shape[1:])}"
            )
        device = self.K_doc.device
        dtype = self.K_doc.dtype
        K_new = K_doc_new.to(device=device, dtype=dtype).unsqueeze(0)
        V_new = V_doc_new.to(device=device, dtype=dtype).unsqueeze(0)
        KR_new = KR_doc_new.to(device=device, dtype=dtype).unsqueeze(0)

        K_doc = torch.cat([self.K_doc[:position], K_new, self.K_doc[position:]], dim=0)
        V_doc = torch.cat([self.V_doc[:position], V_new, self.V_doc[position:]], dim=0)
        KR_doc = torch.cat([self.KR_doc[:position], KR_new, self.KR_doc[position:]], dim=0)

        new_id_t = torch.tensor([int(new_doc_id)], dtype=self.doc_ids.dtype, device=device)
        doc_ids = torch.cat(
            [self.doc_ids[:position], new_id_t, self.doc_ids[position:]], dim=0,
        )

        n_chunks_per_doc = self.n_chunks_per_doc
        K_flat = K_doc.reshape(-1, *K_doc.shape[2:])
        V_flat = V_doc.reshape(-1, *V_doc.shape[2:])
        KR_flat = KR_doc.reshape(-1, *KR_doc.shape[2:])
        chunk_to_doc = (
            doc_ids.view(-1, 1).expand(-1, n_chunks_per_doc).reshape(-1)
        )
        return EncodedCorpus(
            layer_idx=self.layer_idx,
            K=K_flat, V=V_flat, KR=KR_flat,
            chunk_to_doc=chunk_to_doc,
            K_doc=K_doc, V_doc=V_doc, KR_doc=KR_doc,
            doc_ids=doc_ids,
        )

    def gather_topk_docs(self, doc_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice ``K_doc/V_doc/KR_doc`` along the doc axis.

        Args:
            doc_indices: (B, k) integer tensor of row positions into the
                doc-grouped view. NB these are *positions*, not the global
                ``doc_ids`` — use ``self.doc_ids[doc_indices]`` if you need
                the original corpus IDs.

        Returns:
            (K, V, KR), each shape (B, k, n_chunks_per_doc, n_heads, dim).
        """
        B, k = doc_indices.shape
        flat_idx = doc_indices.reshape(-1)
        K = self.K_doc.index_select(0, flat_idx).reshape(B, k, *self.K_doc.shape[1:])
        V = self.V_doc.index_select(0, flat_idx).reshape(B, k, *self.V_doc.shape[1:])
        KR = self.KR_doc.index_select(0, flat_idx).reshape(B, k, *self.KR_doc.shape[1:])
        return K, V, KR


def load_encoded_corpus(
    out_dir: str | Path,
    layers: Optional[Sequence[int]] = None,
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> dict[int, EncodedCorpus]:
    """Load per-MSA-layer encoded corpus tensors from ``out_dir``.

    The directory is expected to contain ``meta.json`` and one
    ``layer_<NN>.pt`` file per MSA layer (as written by
    :func:`offline_encoder.encode_corpus`).
    """
    out_dir = Path(out_dir)
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json under {out_dir}")
    with open(meta_path) as f:
        meta = json.load(f)

    selected = list(meta["msa_layer_indices"])
    if layers is not None:
        selected = [l for l in selected if l in set(layers)]

    is_int4 = bool(meta.get("int4_quantized", False))
    out: dict[int, EncodedCorpus] = {}
    for layer_idx in selected:
        if is_int4:
            from msa.inference.kv_quantizer import load_layer_int4
            path = out_dir / f"layer_{layer_idx:02d}_int4.pt"
            rec = load_layer_int4(path, device="cpu", dtype=torch.bfloat16)
            K_doc = rec["K"]
            V_doc = rec["V"]
            KR_doc = rec["KR"]
            d = {"doc_offsets": rec["doc_offsets"]}
        else:
            path = out_dir / f"layer_{layer_idx:02d}.pt"
            d = torch.load(path, map_location="cpu", weights_only=True)
            # On-disk shape: (n_docs, n_chunks_per_doc, n_heads, head_dim)
            K_doc = d["K"]
            V_doc = d["V"]
            KR_doc = d["KR"]
        n_docs, n_chunks_per_doc, *_ = K_doc.shape
        K_flat = K_doc.reshape(n_docs * n_chunks_per_doc, *K_doc.shape[2:])
        V_flat = V_doc.reshape(n_docs * n_chunks_per_doc, *V_doc.shape[2:])
        KR_flat = KR_doc.reshape(n_docs * n_chunks_per_doc, *KR_doc.shape[2:])
        doc_ids = d["doc_offsets"]
        chunk_to_doc = doc_ids.view(-1, 1).expand(-1, n_chunks_per_doc).reshape(-1)
        cor = EncodedCorpus(
            layer_idx=layer_idx,
            K=K_flat, V=V_flat, KR=KR_flat,
            chunk_to_doc=chunk_to_doc,
            K_doc=K_doc, V_doc=V_doc, KR_doc=KR_doc,
            doc_ids=doc_ids,
        ).to(device=device, dtype=dtype)
        out[layer_idx] = cor
    return out


# -----------------------------------------------------------------------------
# Routing primitive (no model needed; callable on raw Q_r / K_R tensors)
# -----------------------------------------------------------------------------
def score_query_against_corpus(
    Qr: torch.Tensor,
    KR: torch.Tensor,
    n_router_q_heads: Optional[int] = None,
) -> torch.Tensor:
    """Score one query's per-token Q_r against a corpus K_R bank.

    Mirrors the training-mode formula in ``query_msa_attention`` so that
    a query encoded via the same model produces *identical* top-k whether
    routed against in-batch ``Kr_bar`` (training) or against the offline
    corpus K_R (inference).

    Args:
        Qr: (B, L_q, n_router_q_heads, head_dim) — query router projection.
        KR: (n_chunks_total, n_router_k_heads, head_dim) — corpus router K
            from offline encoding (one chunk per row).
        n_router_q_heads: if K has fewer router heads than Q (GQA-style),
            we expand K's heads to match Q's. Defaults to Qr's head count.

    Returns:
        sim_chunk: (B, n_chunks_total) — per-chunk score after the same
            head-mean → query-amax reduction the model uses internally.
            Use ``.topk(k, dim=-1)`` to pull out the k strongest chunks.
    """
    n_router_q_heads = n_router_q_heads if n_router_q_heads is not None else Qr.shape[-2]
    H_r = KR.shape[-2]
    if H_r != n_router_q_heads:
        rep = n_router_q_heads // H_r
        if rep * H_r != n_router_q_heads:
            raise ValueError(
                f"router_k heads ({H_r}) cannot evenly expand to router_q "
                f"heads ({n_router_q_heads}); rep={n_router_q_heads / H_r}"
            )
        KR = (
            KR.unsqueeze(2)
            .expand(-1, -1, rep, -1)
            .reshape(KR.shape[0], n_router_q_heads, KR.shape[-1])
        )
    Qn = F.normalize(Qr.float(), dim=-1)
    Kn = F.normalize(KR.float(), dim=-1)
    # einsum: query token, query head, dim  vs  chunk, head, dim
    sim = torch.einsum("blhd,chd->blhc", Qn, Kn)
    sim = sim.mean(dim=2)        # average over router heads
    sim = sim.amax(dim=1)        # max over query tokens → (B, n_chunks)
    return sim


def topk_chunks(
    sim: torch.Tensor,
    k: int,
    chunk_to_doc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick top-k chunks per query and resolve their source doc indices.

    Args:
        sim: (B, n_chunks_total) per-chunk score.
        k: number of chunks to keep.
        chunk_to_doc: (n_chunks_total,) source-doc index per chunk.

    Returns:
        top_chunk_idx: (B, k) chunk indices into the corpus.
        top_doc_idx:   (B, k) doc indices the chunks belong to.
    """
    k = min(k, sim.shape[-1])
    vals, idx = sim.topk(k, dim=-1)
    docs = chunk_to_doc[idx.view(-1)].view(idx.shape)
    return idx, docs


def topk_unique_docs(
    sim: torch.Tensor,
    k_docs: int,
    chunk_to_doc: torch.Tensor,
) -> torch.Tensor:
    """Return up to ``k_docs`` unique doc indices, ranked by best-chunk score.

    This collapses the chunk axis into the parent doc axis the way the
    EverMind benchmark scoring expects (``[doc_id]`` markers in the LM
    output operate on doc indices, not chunk indices).

    Args:
        sim: (B, n_chunks_total) per-chunk score.
        k_docs: max number of distinct docs to return per query.
        chunk_to_doc: (n_chunks_total,) doc-id per chunk.

    Returns:
        (B, k_docs) — the doc indices of the top-scoring chunks, deduped
        in order. Pads with -1 if fewer than k_docs exist.
    """
    B = sim.shape[0]
    out = torch.full((B, k_docs), -1, dtype=torch.long, device=sim.device)
    # Sort all chunks per row by descending score
    sorted_idx = sim.argsort(dim=-1, descending=True)
    for b in range(B):
        seen: dict[int, None] = {}
        write = 0
        for ci in sorted_idx[b].tolist():
            d = int(chunk_to_doc[ci].item())
            if d in seen:
                continue
            seen[d] = None
            out[b, write] = d
            write += 1
            if write >= k_docs:
                break
    return out
