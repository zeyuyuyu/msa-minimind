"""
Memory Sparse Attention (MSA) model built on top of MiniMind backbone.

Implements the architecture described in "MSA: Memory Sparse Attention for Efficient
End-to-End Memory Model Scaling to 100M Tokens" for the Continual Pre-training stage:
  * Document-wise RoPE: each doc gets position ids starting at 0.
  * Global RoPE for the query, offset by the number of top-k retrieved compressed chunks.
  * Router K / Q projectors produce a specialized routing key/query.
  * Chunk-wise mean pooling on (K, V, K_R) of each doc.
  * Top-k document selection via cosine similarity of routing vectors.
  * Sparse attention from the query Q over [top-k compressed K; local K_query].
  * Routing applied only to the later half of the model's layers.
  * Auxiliary supervised-contrastive (InfoNCE) loss for layer-wise router supervision.

This file only implements the model; data pipeline and CPT trainer live alongside.
"""

from __future__ import annotations
import math
import sys
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import ModelOutput

# Reuse MiniMind primitives.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "minimind")))
from model.model_minimind import (
    MiniMindConfig,
    RMSNorm,
    FeedForward,
    precompute_freqs_cis,
    apply_rotary_pos_emb,
    repeat_kv,
)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
class MSAConfig(MiniMindConfig):
    """Adds MSA-specific knobs on top of MiniMind."""
    model_type = "msa"

    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        msa_start_layer: Optional[int] = None,
        msa_chunk_size: int = 64,
        msa_top_k: int = 4,
        msa_aux_tau: float = 0.07,
        msa_aux_coef: float = 1.0,
        msa_num_router_heads: Optional[int] = None,
        msa_router_k_uses_kv_heads: bool = True,
        msa_max_docs: int = 64,
        **kwargs,
    ):
        super().__init__(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, **kwargs)
        # Apply routing in the later half by default (inclusive from this index).
        self.msa_start_layer = msa_start_layer if msa_start_layer is not None else num_hidden_layers // 2
        self.msa_chunk_size = msa_chunk_size
        self.msa_top_k = msa_top_k
        self.msa_aux_tau = msa_aux_tau
        self.msa_aux_coef = msa_aux_coef
        # Router-Q head count (defaults to num_attention_heads, matching Q).
        # Router-K follows GQA: uses num_key_value_heads per the official reference
        # implementation (EverMind-AI/MSA), aligning with how K is projected in Qwen3.
        self.msa_num_router_heads = msa_num_router_heads if msa_num_router_heads is not None else self.num_attention_heads
        self.msa_router_k_uses_kv_heads = msa_router_k_uses_kv_heads
        # Max docs per sample (used for RoPE budget planning — query offset uses up to top_k slots).
        self.msa_max_docs = msa_max_docs


# --------------------------------------------------------------------------------------
# MSA attention
# --------------------------------------------------------------------------------------
class MSAAttention(nn.Module):
    """A MiniMind-style attention module augmented with router projectors.

    For non-MSA layers the router projectors are unused (created anyway, with a small footprint).
    """

    def __init__(self, layer_id: int, config: MSAConfig):
        super().__init__()
        self.layer_id = layer_id
        self.is_msa = layer_id >= config.msa_start_layer
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        # Router-Q uses `msa_num_router_heads` (default = Q-head count).
        # Router-K follows GQA convention (reference impl uses num_key_value_heads).
        self.n_router_q_heads = config.msa_num_router_heads
        self.n_router_k_heads = (self.num_key_value_heads
                                 if getattr(config, "msa_router_k_uses_kv_heads", True)
                                 else config.msa_num_router_heads)
        self.chunk_size = config.msa_chunk_size

        self.q_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.is_msa:
            # Decoupled router projectors (see reference impl EverMind-AI/MSA: decouple_router=True).
            # router_k uses num_key_value_heads (GQA-aligned), router_q uses num_attention_heads.
            self.kr_proj = nn.Linear(config.hidden_size, self.n_router_k_heads * self.head_dim, bias=False)
            self.qr_proj = nn.Linear(config.hidden_size, self.n_router_q_heads * self.head_dim, bias=False)
            # Init router projectors with small variance for stable warmup.
            nn.init.normal_(self.kr_proj.weight, std=0.01)
            nn.init.normal_(self.qr_proj.weight, std=0.01)
        else:
            self.kr_proj = None
            self.qr_proj = None

        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _chunked_mean(x: torch.Tensor, chunk: int, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean-pool over contiguous chunks of length `chunk` along the sequence dim.

        x: [*, L, ...] with seq dim = -3 (i.e. shape [B, L, H, D]).
        valid_mask: [B, L] float (1 = real, 0 = pad). Used to zero-out pad tokens and renormalize.
        Returns [B, C, H, D] with C = ceil(L / chunk).
        """
        B, L, H, D = x.shape
        C = math.ceil(L / chunk)
        pad_L = C * chunk - L
        if pad_L > 0:
            pad_x = x.new_zeros(B, pad_L, H, D)
            x = torch.cat([x, pad_x], dim=1)
            if valid_mask is not None:
                pad_m = valid_mask.new_zeros(B, pad_L)
                valid_mask = torch.cat([valid_mask, pad_m], dim=1)
        x = x.view(B, C, chunk, H, D)
        if valid_mask is not None:
            m = valid_mask.view(B, C, chunk, 1, 1).to(x.dtype)
            summed = (x * m).sum(dim=2)
            denom = m.sum(dim=2).clamp_min(1.0)
            out = summed / denom
        else:
            out = x.mean(dim=2)
        return out  # [B, C, H, D]

    def pool_doc_kvr(self, doc_hidden: torch.Tensor, doc_pos_emb, doc_valid_mask: Optional[torch.Tensor]):
        """Compute compressed K̄, V̄, K̄_R from per-doc hidden states.

        doc_hidden: [B*N, L_d, D]
        doc_valid_mask: [B*N, L_d] 0/1 (1 = real token)
        Returns K̄ [B*N, C, H_kv, D_h], V̄ [B*N, C, H_kv, D_h], K̄_R [B*N, C, H_r, D_h].
        """
        assert self.is_msa, "pool_doc_kvr only valid for MSA layers"
        BN, L_d, _ = doc_hidden.shape
        xk = self.k_proj(doc_hidden).view(BN, L_d, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(doc_hidden).view(BN, L_d, self.n_local_kv_heads, self.head_dim)
        xkr = self.kr_proj(doc_hidden).view(BN, L_d, self.n_router_k_heads, self.head_dim)
        xk = self.k_norm(xk)
        # Apply doc-wise RoPE to K (each doc uses positions 0..L_d-1).
        cos, sin = doc_pos_emb  # shapes [L_d, head_dim]
        _, xk = apply_rotary_pos_emb(xk.new_zeros(xk.shape), xk, cos, sin)
        # Chunk-wise mean pooling along sequence.
        K_bar = self._chunked_mean(xk, self.chunk_size, doc_valid_mask)
        V_bar = self._chunked_mean(xv, self.chunk_size, doc_valid_mask)
        Kr_bar = self._chunked_mean(xkr, self.chunk_size, doc_valid_mask)
        return K_bar, V_bar, Kr_bar

    # ------------------------------------------------------------------ self-attn paths
    def _self_attention(
        self,
        x: torch.Tensor,
        position_embeddings,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard causal self-attention (used by docs at all layers and by query at non-MSA layers).

        Builds an explicit additive mask that fuses causal + key-pad masking so we never pass
        both ``is_causal=True`` and an ``attn_mask`` to SDPA (which produces implementation-
        dependent behaviour on some backends).
        """
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        xq_t = xq.transpose(1, 2)                                          # [B, H, L, D]
        xk_t = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv_t = repeat_kv(xv, self.n_rep).transpose(1, 2)

        if pad_mask is None:
            out = F.scaled_dot_product_attention(
                xq_t, xk_t, xv_t,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Fuse causal + key-pad into one additive mask.
            neg_inf = torch.finfo(xq_t.dtype).min
            causal = torch.triu(torch.full((seq_len, seq_len), neg_inf, dtype=xq_t.dtype, device=xq_t.device), diagonal=1)
            key_bias = (1.0 - pad_mask.to(xq_t.dtype))[:, None, None, :] * neg_inf   # [B,1,1,L]
            attn_bias = causal[None, None, :, :] + key_bias                           # [B,1,L,L]
            out = F.scaled_dot_product_attention(
                xq_t, xk_t, xv_t,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(out))

    def _query_msa_attention(
        self,
        x: torch.Tensor,
        position_embeddings,
        pooled_KVR,
        pos_doc_labels: Optional[torch.Tensor],
        aux_tau: float,
        query_pad_mask: Optional[torch.Tensor] = None,
    ):
        """Query path in an MSA layer.

        x: [B, L_q, D] -- query hidden states.
        position_embeddings: global RoPE for the query (already offset by top-k outside).
        pooled_KVR: tuple (K̄, V̄, K̄_R) with shapes
            K̄, V̄:  [B, N, C, H_kv, D_h]
            K̄_R:  [B, N, C, H_r,  D_h]
        pos_doc_labels: [B, N] in {0,1}; a doc may be a positive for multiple queries.
            None => skip aux loss.
        query_pad_mask: [B, L_q] with 1=real token, 0=pad. Applied to the local-K portion.
        """
        B, L_q, _ = x.shape
        K_bar, V_bar, Kr_bar = pooled_KVR
        _, N, C, H_kv, D_h = K_bar.shape
        H_r = Kr_bar.shape[-2]

        # Projections for the query.
        xq = self.q_proj(x).view(B, L_q, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(B, L_q, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(B, L_q, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # Routing: QR of query tokens cosine-sim with K̄_R of all doc chunks.
        # router_q may have more heads than router_k (Q-head vs KV-head count under GQA);
        # expand router_k to match router_q heads via `repeat_kv`-style broadcast.
        Qr = self.qr_proj(x).view(B, L_q, self.n_router_q_heads, D_h)
        if H_r != self.n_router_q_heads:
            rep = self.n_router_q_heads // H_r
            Kr_bar_q = Kr_bar.unsqueeze(4).expand(-1, -1, -1, -1, rep, -1).reshape(
                B, N, C, self.n_router_q_heads, D_h)
        else:
            Kr_bar_q = Kr_bar
        Qr_n = F.normalize(Qr.float(), dim=-1)
        Kr_n = F.normalize(Kr_bar_q.float(), dim=-1)
        # Eq. (2): S_ij = max_token mean_head cos(Qr_t, K̄R_ij)
        #    cos_sim: [B, L_q, H_q, N, C]
        cos_sim = torch.einsum("blhd,bnchd->blhnc", Qr_n, Kr_n)
        sim_chunk = cos_sim.mean(dim=2)                     # mean over heads   [B, L_q, N, C]
        # If a query pad mask is given, mask pad tokens out of the "max over tokens" aggregation.
        if query_pad_mask is not None:
            qmask = query_pad_mask.to(sim_chunk.dtype)                 # [B, L_q]
            sim_chunk = sim_chunk.masked_fill(qmask[:, :, None, None] == 0, float("-inf"))
        sim_chunk = sim_chunk.amax(dim=1)                   # max over tokens   [B, N, C]
        s_per_doc = sim_chunk.amax(dim=-1)                  # max over chunks   [B, N]

        # Aux contrastive loss supervising this layer's router.
        aux_loss = x.new_zeros(())
        if pos_doc_labels is not None:
            aux_loss = _supervised_contrastive_loss(s_per_doc, pos_doc_labels, aux_tau)

        # Top-k doc selection.
        top_k_real = min(self.top_k, N)
        _, topk_idx = s_per_doc.topk(top_k_real, dim=-1)                 # [B, k]
        # Gather compressed KVs for the selected docs.
        gather_kv = topk_idx[:, :, None, None, None].expand(-1, -1, C, H_kv, D_h)
        K_topk = torch.gather(K_bar, 1, gather_kv)                        # [B, k, C, H_kv, D_h]
        V_topk = torch.gather(V_bar, 1, gather_kv)
        kv_prefix = top_k_real * C
        K_topk = K_topk.reshape(B, kv_prefix, H_kv, D_h)
        V_topk = V_topk.reshape(B, kv_prefix, H_kv, D_h)

        # Concatenate: Kctx = [K̄_topk ; Kq].
        Kctx = torch.cat([K_topk, xk], dim=1)                             # [B, kv_prefix+L_q, H_kv, D_h]
        Vctx = torch.cat([V_topk, xv], dim=1)

        # Attention. The first kv_prefix positions are cross-attn (all query tokens see all).
        # The last L_q positions are causal within the query (and mask out pad keys).
        xq_t = xq.transpose(1, 2)                                         # [B, H, L_q, D]
        Kctx_t = repeat_kv(Kctx, self.n_rep).transpose(1, 2)              # [B, H, kv_total, D]
        Vctx_t = repeat_kv(Vctx, self.n_rep).transpose(1, 2)

        total_kv = kv_prefix + L_q
        neg_inf = torch.finfo(xq_t.dtype).min
        # Base mask: keep-all for prefix KVs, causal for local KV.
        causal = torch.triu(torch.full((L_q, L_q), neg_inf, dtype=xq_t.dtype, device=xq_t.device), diagonal=1)
        prefix_bias = torch.zeros(L_q, kv_prefix, dtype=xq_t.dtype, device=xq_t.device)
        base_bias = torch.cat([prefix_bias, causal], dim=1)                # [L_q, total_kv]
        attn_bias = base_bias[None, None, :, :].expand(B, 1, L_q, total_kv).contiguous()
        # Mask pad tokens in the local-K segment (columns kv_prefix..) so the query cannot key to them.
        if query_pad_mask is not None:
            key_pad = (1.0 - query_pad_mask.to(xq_t.dtype))[:, None, None, :] * neg_inf   # [B,1,1,L_q]
            # Only apply to the local-K part (trailing L_q columns).
            attn_bias[:, :, :, kv_prefix:] = attn_bias[:, :, :, kv_prefix:] + key_pad
        out = F.scaled_dot_product_attention(
            xq_t, Kctx_t, Vctx_t,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, L_q, -1)
        out = self.resid_dropout(self.o_proj(out))
        return out, aux_loss, s_per_doc

    @property
    def top_k(self) -> int:
        return self._top_k

    def set_top_k(self, k: int):
        self._top_k = k


# --------------------------------------------------------------------------------------
# Aux loss (supervised contrastive, see paper Eq. 5)
# --------------------------------------------------------------------------------------
def _supervised_contrastive_loss(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    scores: [B, N] similarity scores per doc.
    pos_mask: [B, N] 0/1; 1 = positive.

    Implements Eq. (5):
      L = -(1/|P|) sum_{i in P} log( exp(s_i/τ) / (exp(s_i/τ) + sum_{j in N} exp(s_j/τ)) )
    In batched form this becomes a per-positive softmax loss where the positive is pitted
    against all negatives in the same sample.
    """
    s = scores.float() / max(tau, 1e-6)
    pos = pos_mask.float()
    neg = (1.0 - pos)
    # lse over negatives only. Rows with zero negatives get -inf; we mask them out later.
    neg_s = s.masked_fill(neg == 0, float("-inf"))
    lse_neg = torch.logsumexp(neg_s, dim=-1, keepdim=True)               # [B, 1]
    # For each entry we compute log(exp(s) + exp(lse_neg)) - s (the per-position NCE loss).
    denom = torch.logaddexp(s, lse_neg.expand_as(s))
    per = denom - s                                                       # [B, N]
    # Only keep positives and valid rows (≥1 pos, ≥1 neg).
    has_pos = (pos.sum(dim=-1) > 0)
    has_neg = (neg.sum(dim=-1) > 0)
    valid = has_pos & has_neg                                             # [B]
    per_valid = per * pos
    loss_sum = (per_valid * valid[:, None].to(per.dtype)).sum()
    count = (pos * valid[:, None].to(pos.dtype)).sum().clamp_min(1.0)
    return loss_sum / count


# --------------------------------------------------------------------------------------
# Block
# --------------------------------------------------------------------------------------
class MSABlock(nn.Module):
    def __init__(self, layer_id: int, config: MSAConfig):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.self_attn = MSAAttention(layer_id, config)
        self.self_attn.set_top_k(config.msa_top_k)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    @property
    def is_msa(self) -> bool:
        return self.self_attn.is_msa

    # -- doc path (plain causal self-attention per doc)
    def forward_doc(self, hidden_states, position_embeddings, pad_mask=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        out = self.self_attn._self_attention(h, position_embeddings, pad_mask=pad_mask)
        hidden_states = residual + out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    # -- query path at non-MSA layer (plain causal self-attention)
    def forward_query_self(self, hidden_states, position_embeddings, pad_mask=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        out = self.self_attn._self_attention(h, position_embeddings, pad_mask=pad_mask)
        hidden_states = residual + out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    # -- query path at MSA layer (sparse cross-attn to pooled doc KVs + local causal)
    def forward_query_msa(self, hidden_states, position_embeddings, pooled_KVR, pos_doc_labels, aux_tau, query_pad_mask=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        out, aux_loss, s_per_doc = self.self_attn._query_msa_attention(
            h, position_embeddings, pooled_KVR, pos_doc_labels, aux_tau, query_pad_mask=query_pad_mask,
        )
        hidden_states = residual + out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, aux_loss, s_per_doc

    # -- helper: compute pooled K̄, V̄, K̄_R from doc hidden states entering this MSA layer
    def pool_doc_kvr(self, doc_hidden, doc_pos_emb, doc_valid_mask):
        assert self.is_msa, "pool_doc_kvr only valid for MSA layers"
        h = self.input_layernorm(doc_hidden)
        return self.self_attn.pool_doc_kvr(h, doc_pos_emb, doc_valid_mask)


# --------------------------------------------------------------------------------------
# Full MSA model
# --------------------------------------------------------------------------------------
@dataclass
class MSAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    routing_scores: Optional[torch.FloatTensor] = None            # last MSA layer
    per_layer_routing: Optional[dict] = None                      # {layer_idx: [B, N]}


class MSAModel(nn.Module):
    def __init__(self, config: MSAConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MSABlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _rope_slice(self, start: int, end: int, device):
        return (self.freqs_cos[start:end].to(device), self.freqs_sin[start:end].to(device))

    def forward(
        self,
        doc_input_ids: torch.Tensor,            # [B, N, L_d]
        query_input_ids: torch.Tensor,          # [B, L_q]
        doc_attention_mask: Optional[torch.Tensor] = None,   # [B, N, L_d]
        query_attention_mask: Optional[torch.Tensor] = None,  # [B, L_q]
        pos_doc_labels: Optional[torch.Tensor] = None,        # [B, N] 0/1
        labels: Optional[torch.Tensor] = None,                 # [B, L_q] with -100 ignored
    ):
        B, N, L_d = doc_input_ids.shape
        _, L_q = query_input_ids.shape
        device = doc_input_ids.device
        # ------------------------------------------------------------------ doc encoding
        h_d = self.embed_tokens(doc_input_ids.reshape(B * N, L_d))           # [B*N, L_d, D]
        h_d = self.dropout(h_d)
        doc_pos_emb = self._rope_slice(0, L_d, device)
        doc_pad = doc_attention_mask.reshape(B * N, L_d).to(h_d.dtype) if doc_attention_mask is not None else None

        # Cache of pooled doc KV per MSA layer (indexed by layer id).
        pooled_cache = {}

        for i, layer in enumerate(self.layers):
            # Before processing this layer for docs, if it's an MSA layer, snapshot the pooled K̄/V̄/K̄_R
            # using the CURRENT doc hidden state (= input to layer i). This matches the paper: the MSA
            # layer computes K,V,K_R from the doc hidden state entering that layer.
            if layer.is_msa:
                K_bar, V_bar, Kr_bar = layer.pool_doc_kvr(h_d, doc_pos_emb, doc_pad)
                BN = K_bar.shape[0]
                K_bar = K_bar.view(B, N, *K_bar.shape[1:])
                V_bar = V_bar.view(B, N, *V_bar.shape[1:])
                Kr_bar = Kr_bar.view(B, N, *Kr_bar.shape[1:])
                pooled_cache[i] = (K_bar, V_bar, Kr_bar)
            # Now advance the doc hidden states through this layer (independent doc processing).
            h_d = layer.forward_doc(h_d, doc_pos_emb, pad_mask=doc_pad)

        # ------------------------------------------------------------------ query encoding
        h_q = self.embed_tokens(query_input_ids)                              # [B, L_q, D]
        h_q = self.dropout(h_q)
        # Query positions are offset by top_k so the query perceives the retrieved docs as
        # a logical prefix. The compressed doc KVs live at positions [0, top_k) virtually.
        query_offset = self.config.msa_top_k
        q_pos_emb = self._rope_slice(query_offset, query_offset + L_q, device)
        q_pad = query_attention_mask.to(h_q.dtype) if query_attention_mask is not None else None

        aux_loss_sum = h_q.new_zeros(())
        aux_loss_count = 0
        per_layer_routing = {}

        for i, layer in enumerate(self.layers):
            if layer.is_msa and i in pooled_cache:
                h_q, aux_i, s_per_doc = layer.forward_query_msa(
                    h_q, q_pos_emb, pooled_cache[i], pos_doc_labels, self.config.msa_aux_tau,
                    query_pad_mask=q_pad,
                )
                aux_loss_sum = aux_loss_sum + aux_i
                aux_loss_count += 1
                per_layer_routing[i] = s_per_doc
            else:
                h_q = layer.forward_query_self(h_q, q_pos_emb, pad_mask=q_pad)

        h_q = self.norm(h_q)

        aux_loss = aux_loss_sum / max(aux_loss_count, 1)
        # Return the last-layer scores for convenience, plus the per-layer dict.
        last_routing = per_layer_routing[max(per_layer_routing)] if per_layer_routing else None
        return h_q, aux_loss, last_routing, per_layer_routing


class MSAForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MSAConfig

    def __init__(self, config: MSAConfig = None):
        self.config = config or MSAConfig()
        super().__init__(self.config)
        self.model = MSAModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    # Load weights from a pretrained MiniMind model (compatible layer structure).
    def load_minimind_pretrained(self, minimind_state_dict: dict, strict: bool = False):
        """Load backbone weights from a MiniMind checkpoint.

        Router projectors (kr_proj, qr_proj) are freshly initialized and skipped.
        Renames layer sub-module keys if needed.
        """
        own = self.state_dict()
        compat = {}
        missing = []
        for k, v in minimind_state_dict.items():
            if k in own and own[k].shape == v.shape:
                compat[k] = v
            else:
                missing.append(k)
        self.load_state_dict(compat, strict=False)
        return {"loaded": list(compat.keys()), "skipped_from_ckpt": missing}

    def forward(
        self,
        doc_input_ids,
        query_input_ids,
        doc_attention_mask=None,
        query_attention_mask=None,
        pos_doc_labels=None,
        labels=None,
        lm_loss_coef: float = 1.0,
        aux_loss_coef: Optional[float] = None,
        logits_to_keep: int = 0,
    ):
        hidden_states, aux_loss, routing, per_layer = self.model(
            doc_input_ids=doc_input_ids,
            query_input_ids=query_input_ids,
            doc_attention_mask=doc_attention_mask,
            query_attention_mask=query_attention_mask,
            pos_doc_labels=pos_doc_labels,
            labels=labels,
        )
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        lm_loss = None
        if labels is not None:
            # Next-token prediction over the query sequence.
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        total = None
        if lm_loss is not None:
            ac = aux_loss_coef if aux_loss_coef is not None else self.config.msa_aux_coef
            total = lm_loss_coef * lm_loss + ac * aux_loss

        return MSAOutput(
            loss=total,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            logits=logits,
            routing_scores=routing,
            per_layer_routing=per_layer,
        )
