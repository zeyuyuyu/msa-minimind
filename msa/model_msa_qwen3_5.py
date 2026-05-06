"""Memory Sparse Attention (MSA) on top of a Qwen3.5 hybrid backbone.

Qwen3.5 is a *hybrid* architecture: of its 32 decoder layers, 24 are Gated
DeltaNet (linear attention, stateful) and 8 are Gated Attention (standard
causal attention with an extra per-token output gate). MSA can only be layered
on top of the 8 ``full_attention`` layers — the linear-attention layers are
left untouched and run through the transformers reference implementation.

Key differences vs the Qwen3-8B MSA port (``model_msa_qwen3.py``):

* ``head_dim = 256`` (vs 128) and ``attn_output_gate = True`` — the ``q_proj``
  is fat (``heads * head_dim * 2``), with the back half acting as a gate
  applied as ``attn_output * sigmoid(gate)`` before ``o_proj``.
* Partial RoPE (``partial_rotary_factor = 0.25``): only the first
  ``head_dim * 0.25 = 64`` dims of Q/K are rotated; the remaining 192 are
  passed through.
* MRoPE-interleaved RoPE (text branch = T-axis only) — we just reuse the
  reference ``Qwen3_5TextRotaryEmbedding`` / ``apply_rotary_pos_emb``.
* Linear-attention layers are reused verbatim from transformers
  (``Qwen3_5GatedDeltaNet``). Doc encoding runs every doc as an independent
  batch item (no ``cache_params``) so each doc has its own hidden recurrent
  state / conv state.
* Module names / tensor shapes exactly match the reference implementation so
  we can load the pretrained base checkpoint with ``load_state_dict``.

Router projectors (``qr_proj`` / ``kr_proj``) are added *only* on MSA layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import ModelOutput

# Reference primitives from transformers' Qwen3.5 implementation.
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5MLP,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5GatedDeltaNet,
    Qwen3_5DynamicCache,
    apply_rotary_pos_emb as _apply_partial_rope,
    repeat_kv,
)


# ---------------------------------------------------------------------------
# Streaming KV cache for query decode
# ---------------------------------------------------------------------------
class MSAQueryKVCache:
    """Streaming cache used by ``forward_query_step`` to avoid recomputing
    self-attention / linear-attention state every decode step.

    Holds three pieces:

    * **Self-attention KV** — ``self_K[i] / self_V[i]`` of shape
      ``(B, H_kv, L_seen, head_dim)`` for each of the 8 ``full_attention``
      layers (both MSA and the 0 non-MSA full-attn layers in our config).
    * **MSA frozen routing** — for each MSA layer ``i``, the gathered
      ``msa_K_topk[i] / msa_V_topk[i]`` (shape ``(B, top_k*C, H_kv, D)``)
      from the prefill-time routing decision. Decode steps reuse them so
      the router never re-fires.
    * **Linear-attention recurrent state** — wrapped in the upstream
      ``Qwen3_5DynamicCache`` (``conv_states`` / ``recurrent_states``).

    Plus a ``seq_len`` counter that tracks how many query tokens have
    flowed through, so the next slice's RoPE positions land at the right
    offset.
    """

    def __init__(self, config: "MSAQwen3_5Config"):
        self.config = config
        self.linear_cache = Qwen3_5DynamicCache(config)
        self.self_K: dict[int, torch.Tensor] = {}
        self.self_V: dict[int, torch.Tensor] = {}
        self.msa_K_topk: dict[int, torch.Tensor] = {}
        self.msa_V_topk: dict[int, torch.Tensor] = {}
        self.msa_topk_idx: dict[int, torch.Tensor] = {}
        self.last_s_per_doc: dict[int, torch.Tensor] = {}
        self.seq_len: int = 0

    def has_msa_routing(self) -> bool:
        return len(self.msa_topk_idx) > 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class MSAQwen3_5Config(Qwen3_5TextConfig):
    """Adds MSA-specific knobs on top of ``Qwen3_5TextConfig``.

    ``msa_layer_indices`` is the explicit list of layer indices on which to
    inject MSA. Defaults to ``all layers with layer_types[i] == "full_attention"``.
    """

    model_type = "msa_qwen3_5"

    def __init__(
        self,
        msa_layer_indices: Optional[list[int]] = None,
        msa_chunk_size: int = 64,
        msa_top_k: int = 16,
        msa_aux_tau: float = 0.07,
        msa_aux_coef: float = 1.0,
        msa_num_router_heads: Optional[int] = None,
        msa_router_k_uses_kv_heads: bool = True,
        msa_max_docs: int = 64,
        msa_dropout: float = 0.0,
        msa_gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if msa_layer_indices is None:
            msa_layer_indices = [
                i for i, t in enumerate(self.layer_types) if t == "full_attention"
            ]
        else:
            for i in msa_layer_indices:
                assert self.layer_types[i] == "full_attention", (
                    f"MSA can only be placed on full_attention layers, got "
                    f"layer_types[{i}] = {self.layer_types[i]}"
                )
        self.msa_layer_indices = list(msa_layer_indices)
        self.msa_chunk_size = msa_chunk_size
        self.msa_top_k = msa_top_k
        self.msa_aux_tau = msa_aux_tau
        self.msa_aux_coef = msa_aux_coef
        self.msa_num_router_heads = (
            msa_num_router_heads
            if msa_num_router_heads is not None
            else self.num_attention_heads
        )
        self.msa_router_k_uses_kv_heads = msa_router_k_uses_kv_heads
        self.msa_max_docs = msa_max_docs
        self.msa_dropout = msa_dropout
        self.msa_gradient_checkpointing = bool(msa_gradient_checkpointing)


# ---------------------------------------------------------------------------
# Aux loss (supervised contrastive, paper Eq. 5) — identical to Qwen3-8B port.
# ---------------------------------------------------------------------------
def _supervised_contrastive_loss(
    scores: torch.Tensor, pos_mask: torch.Tensor, tau: float
) -> torch.Tensor:
    s = scores.float() / max(tau, 1e-6)
    pos = pos_mask.float()
    neg = 1.0 - pos
    neg_s = s.masked_fill(neg == 0, float("-inf"))
    lse_neg = torch.logsumexp(neg_s, dim=-1, keepdim=True)
    denom = torch.logaddexp(s, lse_neg.expand_as(s))
    per = denom - s
    has_pos = (pos.sum(dim=-1) > 0)
    has_neg = (neg.sum(dim=-1) > 0)
    valid = has_pos & has_neg
    per_valid = per * pos
    loss_sum = (per_valid * valid[:, None].to(per.dtype)).sum()
    count = (pos * valid[:, None].to(pos.dtype)).sum().clamp_min(1.0)
    return loss_sum / count


# ---------------------------------------------------------------------------
# Attention — MSA-aware replacement for ``Qwen3_5Attention`` on full layers.
# ---------------------------------------------------------------------------
class MSAQwen3_5Attention(nn.Module):
    """Gated Attention with optional MSA router projectors.

    Matches ``Qwen3_5Attention`` parameter layout exactly so pretrained
    weights can be loaded by name. Adds ``qr_proj`` / ``kr_proj`` when
    ``is_msa`` is True.
    """

    def __init__(self, layer_idx: int, config: MSAQwen3_5Config, is_msa: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_msa = is_msa
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        attn_bias = config.attention_bias
        # q_proj fat: outputs (heads * head_dim * 2) — last half is the gate.
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=attn_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attn_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attn_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=attn_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Router projectors (only on MSA layers).
        self.n_router_q_heads = config.msa_num_router_heads
        self.n_router_k_heads = (
            self.num_key_value_heads
            if config.msa_router_k_uses_kv_heads
            else config.msa_num_router_heads
        )
        self.chunk_size = config.msa_chunk_size
        if is_msa:
            self.qr_proj = nn.Linear(
                config.hidden_size,
                self.n_router_q_heads * self.head_dim,
                bias=False,
            )
            self.kr_proj = nn.Linear(
                config.hidden_size,
                self.n_router_k_heads * self.head_dim,
                bias=False,
            )
            nn.init.normal_(self.qr_proj.weight, std=0.01)
            nn.init.normal_(self.kr_proj.weight, std=0.01)
        else:
            self.qr_proj = None
            self.kr_proj = None

        self.dropout_p = config.msa_dropout
        self.resid_dropout = (
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )
        self._top_k = config.msa_top_k

    def set_top_k(self, k: int) -> None:
        self._top_k = k

    # --------------------------------------------------------------------- QKV
    def _project_qkv(self, x: torch.Tensor):
        """Run q/k/v projections (with q_norm/k_norm), return in BLHD layout.

        Returns (q_states, k_states, v_states, gate) where gate is BLD_full.
        """
        B, L, _ = x.shape
        q_fat = self.q_proj(x).view(B, L, self.num_attention_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_fat, 2, dim=-1)
        gate = gate.reshape(B, L, -1)  # (B, L, heads * head_dim)

        q = self.q_norm(q)
        k = self.k_norm(
            self.k_proj(x).view(B, L, self.num_key_value_heads, self.head_dim)
        )
        v = self.v_proj(x).view(B, L, self.num_key_value_heads, self.head_dim)
        return q, k, v, gate

    # ---------------------------------------------------------- chunk utility
    @staticmethod
    def _chunked_mean(
        x: torch.Tensor, chunk: int, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean-pool over contiguous chunks along seq dim.

        ``x`` shape: ``[B, L, H, D]``. ``valid_mask`` shape: ``[B, L]``.
        Returns ``[B, C, H, D]``.
        """
        B, L, H, D = x.shape
        C = math.ceil(L / chunk)
        pad_L = C * chunk - L
        if pad_L > 0:
            x = torch.cat([x, x.new_zeros(B, pad_L, H, D)], dim=1)
            if valid_mask is not None:
                valid_mask = torch.cat(
                    [valid_mask, valid_mask.new_zeros(B, pad_L)], dim=1
                )
        x = x.view(B, C, chunk, H, D)
        if valid_mask is not None:
            m = valid_mask.view(B, C, chunk, 1, 1).to(x.dtype)
            summed = (x * m).sum(dim=2)
            denom = m.sum(dim=2).clamp_min(1.0)
            return summed / denom
        return x.mean(dim=2)

    # ------------------------------------------------- doc K̄, V̄, K̄_R pooling
    def pool_doc_kvr(
        self,
        doc_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        doc_valid_mask: Optional[torch.Tensor],
    ):
        """Compute compressed K̄ / V̄ / K̄_R from a doc's hidden states at this layer.

        ``doc_hidden`` is the *pre-layernorm* hidden state; the caller applies
        ``input_layernorm`` first (mirrors MSAQwen3 port).
        """
        assert self.is_msa, "pool_doc_kvr only valid on MSA layers"
        BN, L_d, _ = doc_hidden.shape
        xk = self.k_norm(
            self.k_proj(doc_hidden).view(BN, L_d, self.num_key_value_heads, self.head_dim)
        )
        xv = self.v_proj(doc_hidden).view(
            BN, L_d, self.num_key_value_heads, self.head_dim
        )
        xkr = self.kr_proj(doc_hidden).view(
            BN, L_d, self.n_router_k_heads, self.head_dim
        )

        # Partial RoPE on xk only (xkr is intentionally rope-free — router
        # reads semantic similarity without positional coupling).
        cos, sin = position_embeddings
        dummy_q = xk.new_zeros(xk.shape)
        # apply_rotary_pos_emb expects [B, H, L, D] when unsqueeze_dim=1
        # (default) — we pass BLHD with unsqueeze_dim=2 to match.
        _, xk_rot = _apply_partial_rope(dummy_q, xk, cos, sin, unsqueeze_dim=2)

        K_bar = self._chunked_mean(xk_rot, self.chunk_size, doc_valid_mask)
        V_bar = self._chunked_mean(xv, self.chunk_size, doc_valid_mask)
        Kr_bar = self._chunked_mean(xkr, self.chunk_size, doc_valid_mask)
        return K_bar, V_bar, Kr_bar

    # --------------------------------------------------- plain self-attention
    def self_attention(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard causal gated-attention. Used for docs on every layer and
        for queries on non-MSA layers (which here never fires, since every
        full_attention layer is MSA). Kept for symmetry and for ``pool_doc_kvr``.
        """
        B, L, _ = x.shape
        q, k, v, gate = self._project_qkv(x)

        cos, sin = position_embeddings
        q, k = _apply_partial_rope(q, k, cos, sin, unsqueeze_dim=2)

        q_t = q.transpose(1, 2)  # [B, H, L, D]
        k_t = repeat_kv(k.transpose(1, 2), self.num_key_value_groups)
        v_t = repeat_kv(v.transpose(1, 2), self.num_key_value_groups)

        if pad_mask is None:
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            neg_inf = torch.finfo(q_t.dtype).min
            causal = torch.triu(
                torch.full((L, L), neg_inf, dtype=q_t.dtype, device=q_t.device),
                diagonal=1,
            )
            key_bias = (1.0 - pad_mask.to(q_t.dtype))[:, None, None, :] * neg_inf
            attn_bias = causal[None, None, :, :] + key_bias
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=attn_bias,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )

        out = out.transpose(1, 2).reshape(B, L, -1)
        out = out * torch.sigmoid(gate)
        out = self.o_proj(out)
        return self.resid_dropout(out)

    # ------------------------------------------------------ MSA cross-attention
    def query_msa_attention(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        pooled_KVR,
        pos_doc_labels: Optional[torch.Tensor],
        aux_tau: float,
        query_pad_mask: Optional[torch.Tensor] = None,
    ):
        assert self.is_msa, "query_msa_attention only on MSA layers"
        B, L_q, _ = x.shape
        K_bar, V_bar, Kr_bar = pooled_KVR
        _, N, C, H_kv, D_h = K_bar.shape
        H_r = Kr_bar.shape[-2]

        q, k, v, gate = self._project_qkv(x)
        cos, sin = position_embeddings
        q, k = _apply_partial_rope(q, k, cos, sin, unsqueeze_dim=2)

        # Router Qr (no RoPE).
        Qr = self.qr_proj(x).view(B, L_q, self.n_router_q_heads, D_h)
        if H_r != self.n_router_q_heads:
            rep = self.n_router_q_heads // H_r
            Kr_bar_q = (
                Kr_bar.unsqueeze(4)
                .expand(-1, -1, -1, -1, rep, -1)
                .reshape(B, N, C, self.n_router_q_heads, D_h)
            )
        else:
            Kr_bar_q = Kr_bar
        Qr_n = F.normalize(Qr.float(), dim=-1)
        Kr_n = F.normalize(Kr_bar_q.float(), dim=-1)
        cos_sim = torch.einsum("blhd,bnchd->blhnc", Qr_n, Kr_n)
        sim_chunk = cos_sim.mean(dim=2)  # average over router heads
        if query_pad_mask is not None:
            qmask = query_pad_mask.to(sim_chunk.dtype)
            sim_chunk = sim_chunk.masked_fill(qmask[:, :, None, None] == 0, float("-inf"))
        sim_chunk = sim_chunk.amax(dim=1)       # over query positions
        s_per_doc = sim_chunk.amax(dim=-1)      # over chunks → [B, N]

        aux_loss = x.new_zeros(())
        if pos_doc_labels is not None:
            aux_loss = _supervised_contrastive_loss(s_per_doc, pos_doc_labels, aux_tau)

        top_k_real = min(self._top_k, N)
        _, topk_idx = s_per_doc.topk(top_k_real, dim=-1)
        gather_kv = topk_idx[:, :, None, None, None].expand(-1, -1, C, H_kv, D_h)
        K_topk = torch.gather(K_bar, 1, gather_kv)
        V_topk = torch.gather(V_bar, 1, gather_kv)
        kv_prefix = top_k_real * C
        K_topk = K_topk.reshape(B, kv_prefix, H_kv, D_h)
        V_topk = V_topk.reshape(B, kv_prefix, H_kv, D_h)

        K_ctx = torch.cat([K_topk, k], dim=1)
        V_ctx = torch.cat([V_topk, v], dim=1)

        q_t = q.transpose(1, 2)
        K_t = repeat_kv(K_ctx.transpose(1, 2), self.num_key_value_groups)
        V_t = repeat_kv(V_ctx.transpose(1, 2), self.num_key_value_groups)

        total_kv = kv_prefix + L_q
        neg_inf = torch.finfo(q_t.dtype).min
        causal = torch.triu(
            torch.full((L_q, L_q), neg_inf, dtype=q_t.dtype, device=q_t.device),
            diagonal=1,
        )
        prefix_bias = torch.zeros(L_q, kv_prefix, dtype=q_t.dtype, device=q_t.device)
        base_bias = torch.cat([prefix_bias, causal], dim=1)
        attn_bias = base_bias[None, None, :, :].expand(B, 1, L_q, total_kv).contiguous()
        if query_pad_mask is not None:
            key_pad = (1.0 - query_pad_mask.to(q_t.dtype))[:, None, None, :] * neg_inf
            attn_bias[:, :, :, kv_prefix:] = attn_bias[:, :, :, kv_prefix:] + key_pad

        out = F.scaled_dot_product_attention(
            q_t, K_t, V_t,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).reshape(B, L_q, -1)
        out = out * torch.sigmoid(gate)
        out = self.o_proj(out)
        return self.resid_dropout(out), aux_loss, s_per_doc

    # ----------------------------------------- KV-cached self-attention
    def self_attention_cached(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_K: Optional[torch.Tensor] = None,
        past_V: Optional[torch.Tensor] = None,
    ):
        """KV-cached gated self-attention. Used for ``full_attention`` layers
        that are NOT MSA (rare in our config). ``position_embeddings`` only
        cover the new ``L`` tokens; previously-seen K is already RoPE'd.
        Returns ``(output, K_full, V_full)``.
        """
        B, L, _ = x.shape
        q, k, v, gate = self._project_qkv(x)
        cos, sin = position_embeddings
        q, k = _apply_partial_rope(q, k, cos, sin, unsqueeze_dim=2)

        q_t = q.transpose(1, 2)
        k_new = k.transpose(1, 2)
        v_new = v.transpose(1, 2)

        if past_K is not None:
            K_full = torch.cat([past_K, k_new], dim=2)
            V_full = torch.cat([past_V, v_new], dim=2)
        else:
            K_full = k_new
            V_full = v_new

        K_t = repeat_kv(K_full, self.num_key_value_groups)
        V_t = repeat_kv(V_full, self.num_key_value_groups)

        L_total = K_full.shape[2]
        L_offset = L_total - L

        if L > 1:
            neg_inf = torch.finfo(q_t.dtype).min
            causal_self = torch.triu(
                torch.full((L, L), neg_inf, dtype=q_t.dtype, device=q_t.device),
                diagonal=1,
            )
            prefix_zero = torch.zeros(
                (L, L_offset), dtype=q_t.dtype, device=q_t.device,
            )
            attn_bias = torch.cat([prefix_zero, causal_self], dim=1)[None, None, :, :]
            out = F.scaled_dot_product_attention(q_t, K_t, V_t, attn_mask=attn_bias)
        else:
            out = F.scaled_dot_product_attention(q_t, K_t, V_t, is_causal=False)

        out = out.transpose(1, 2).reshape(B, L, -1) * torch.sigmoid(gate)
        out = self.o_proj(out)
        return self.resid_dropout(out), K_full, V_full

    # ----------------------------------- KV-cached MSA cross-attention
    def query_msa_attention_cached(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        pooled_KVR,
        past_K: Optional[torch.Tensor] = None,
        past_V: Optional[torch.Tensor] = None,
        fixed_K_topk: Optional[torch.Tensor] = None,
        fixed_V_topk: Optional[torch.Tensor] = None,
        fixed_topk_idx: Optional[torch.Tensor] = None,
    ):
        """KV-cached MSA cross-attention.

        Two modes:

        * **Prefill** (``fixed_*=None``) — runs the router on the new
          ``L_q`` tokens, picks top-k docs, gathers ``K_topk / V_topk`` for
          downstream reuse.
        * **Decode** (``fixed_*`` provided) — skips the router; uses the
          cached top-k corpus tensors as-is.

        Returns ``(output, s_per_doc, K_self_full, V_self_full,
        K_topk, V_topk, topk_idx)``. ``s_per_doc`` is ``None`` in decode
        mode (caller can pull the cached prefill value if needed).
        """
        assert self.is_msa, "query_msa_attention_cached only on MSA layers"
        B, L_q, _ = x.shape
        K_bar, V_bar, Kr_bar = pooled_KVR
        _, N, C, H_kv, D_h = K_bar.shape
        H_r = Kr_bar.shape[-2]

        q, k, v, gate = self._project_qkv(x)
        cos, sin = position_embeddings
        q, k = _apply_partial_rope(q, k, cos, sin, unsqueeze_dim=2)

        if fixed_K_topk is None:
            Qr = self.qr_proj(x).view(B, L_q, self.n_router_q_heads, D_h)
            if H_r != self.n_router_q_heads:
                rep = self.n_router_q_heads // H_r
                Kr_bar_q = (
                    Kr_bar.unsqueeze(4)
                    .expand(-1, -1, -1, -1, rep, -1)
                    .reshape(B, N, C, self.n_router_q_heads, D_h)
                )
            else:
                Kr_bar_q = Kr_bar
            Qr_n = F.normalize(Qr.float(), dim=-1)
            Kr_n = F.normalize(Kr_bar_q.float(), dim=-1)
            cos_sim = torch.einsum("blhd,bnchd->blhnc", Qr_n, Kr_n)
            sim_chunk = cos_sim.mean(dim=2)
            sim_chunk = sim_chunk.amax(dim=1)
            s_per_doc = sim_chunk.amax(dim=-1)

            top_k_real = min(self._top_k, N)
            _, topk_idx = s_per_doc.topk(top_k_real, dim=-1)
            gather_kv = topk_idx[:, :, None, None, None].expand(-1, -1, C, H_kv, D_h)
            K_topk = torch.gather(K_bar, 1, gather_kv)
            V_topk = torch.gather(V_bar, 1, gather_kv)
            kv_prefix = top_k_real * C
            K_topk = K_topk.reshape(B, kv_prefix, H_kv, D_h)
            V_topk = V_topk.reshape(B, kv_prefix, H_kv, D_h)
        else:
            K_topk = fixed_K_topk
            V_topk = fixed_V_topk
            topk_idx = fixed_topk_idx
            kv_prefix = K_topk.shape[1]
            s_per_doc = None

        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        if past_K is not None:
            K_self = torch.cat([past_K, k_t], dim=2)
            V_self = torch.cat([past_V, v_t], dim=2)
        else:
            K_self = k_t
            V_self = v_t

        L_self_total = K_self.shape[2]
        L_offset = L_self_total - L_q

        K_topk_t = K_topk.transpose(1, 2)
        V_topk_t = V_topk.transpose(1, 2)
        K_ctx_t = torch.cat([K_topk_t, K_self], dim=2)
        V_ctx_t = torch.cat([V_topk_t, V_self], dim=2)
        K_t_full = repeat_kv(K_ctx_t, self.num_key_value_groups)
        V_t_full = repeat_kv(V_ctx_t, self.num_key_value_groups)

        q_t = q.transpose(1, 2)

        if L_q > 1:
            neg_inf = torch.finfo(q_t.dtype).min
            causal_self = torch.triu(
                torch.full((L_q, L_q), neg_inf, dtype=q_t.dtype, device=q_t.device),
                diagonal=1,
            )
            prefix_zeros = torch.zeros(
                (L_q, kv_prefix + L_offset), dtype=q_t.dtype, device=q_t.device,
            )
            attn_bias = torch.cat([prefix_zeros, causal_self], dim=1)[None, None, :, :]
            out = F.scaled_dot_product_attention(q_t, K_t_full, V_t_full, attn_mask=attn_bias)
        else:
            out = F.scaled_dot_product_attention(q_t, K_t_full, V_t_full, is_causal=False)

        out = out.transpose(1, 2).reshape(B, L_q, -1) * torch.sigmoid(gate)
        out = self.o_proj(out)
        return self.resid_dropout(out), s_per_doc, K_self, V_self, K_topk, V_topk, topk_idx


# ---------------------------------------------------------------------------
# DecoderLayer (dispatches to DeltaNet or MSAAttention by layer_type)
# ---------------------------------------------------------------------------
class MSAQwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: MSAQwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_msa_layer = layer_idx in config.msa_layer_indices

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = MSAQwen3_5Attention(layer_idx, config, self.is_msa_layer)
        else:
            raise ValueError(f"unknown layer_type {self.layer_type}")

        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @property
    def is_full_attention(self) -> bool:
        return self.layer_type == "full_attention"

    # --------------------------------------------------------- doc forward
    def forward_doc(self, hidden_states, position_embeddings, attention_mask=None):
        """Forward a doc (or doc batch) through this layer.

        For MSA layers the caller has already pooled K̄/V̄/K̄_R before this — but
        we still need to let the full-attention self-attention consume the doc
        so subsequent layers see the correct hidden states. For linear_attention
        layers, ``cache_params`` is None so each doc gets an independent
        recurrent / conv state.
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            h = self.linear_attn(
                hidden_states=h,
                cache_params=None,
                cache_position=None,
                attention_mask=attention_mask,
            )
        else:
            h = self.self_attn.self_attention(h, position_embeddings, pad_mask=attention_mask)
        hidden_states = residual + h
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    # ----------------------------- MSA pooling BEFORE doc traverses this layer
    def pool_doc_kvr(self, hidden_states, position_embeddings, attention_mask=None):
        assert self.is_full_attention and self.is_msa_layer
        h = self.input_layernorm(hidden_states)
        return self.self_attn.pool_doc_kvr(h, position_embeddings, attention_mask)

    # -------------------------------------------------------- query forward
    def forward_query_self(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            h = self.linear_attn(
                hidden_states=h,
                cache_params=None,
                cache_position=None,
                attention_mask=attention_mask,
            )
        else:
            h = self.self_attn.self_attention(h, position_embeddings, pad_mask=attention_mask)
        hidden_states = residual + h
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    def forward_query_msa(
        self,
        hidden_states,
        position_embeddings,
        pooled_KVR,
        pos_doc_labels,
        aux_tau,
        attention_mask=None,
    ):
        assert self.is_full_attention and self.is_msa_layer
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        out, aux_loss, s_per_doc = self.self_attn.query_msa_attention(
            h, position_embeddings, pooled_KVR, pos_doc_labels, aux_tau,
            query_pad_mask=attention_mask,
        )
        hidden_states = residual + out
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, aux_loss, s_per_doc

    # --------------------------------- KV-cached query forward variants
    def forward_query_self_cached(
        self,
        hidden_states,
        position_embeddings,
        cache: "MSAQueryKVCache",
        cache_position: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            h = self.linear_attn(
                hidden_states=h,
                cache_params=cache.linear_cache,
                cache_position=cache_position,
                attention_mask=None,
            )
        else:
            past_K = cache.self_K.get(self.layer_idx)
            past_V = cache.self_V.get(self.layer_idx)
            h, K_full, V_full = self.self_attn.self_attention_cached(
                h, position_embeddings,
                past_K=past_K, past_V=past_V,
            )
            cache.self_K[self.layer_idx] = K_full
            cache.self_V[self.layer_idx] = V_full
        hidden_states = residual + h
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    def forward_query_msa_cached(
        self,
        hidden_states,
        position_embeddings,
        pooled_KVR,
        cache: "MSAQueryKVCache",
    ):
        assert self.is_full_attention and self.is_msa_layer
        residual = hidden_states
        h = self.input_layernorm(hidden_states)

        past_K = cache.self_K.get(self.layer_idx)
        past_V = cache.self_V.get(self.layer_idx)
        fixed_K_topk = cache.msa_K_topk.get(self.layer_idx)
        fixed_V_topk = cache.msa_V_topk.get(self.layer_idx)
        fixed_topk_idx = cache.msa_topk_idx.get(self.layer_idx)

        out, s_per_doc, K_self, V_self, K_topk, V_topk, topk_idx = (
            self.self_attn.query_msa_attention_cached(
                h, position_embeddings, pooled_KVR,
                past_K=past_K, past_V=past_V,
                fixed_K_topk=fixed_K_topk,
                fixed_V_topk=fixed_V_topk,
                fixed_topk_idx=fixed_topk_idx,
            )
        )
        cache.self_K[self.layer_idx] = K_self
        cache.self_V[self.layer_idx] = V_self
        if fixed_K_topk is None:
            cache.msa_K_topk[self.layer_idx] = K_topk
            cache.msa_V_topk[self.layer_idx] = V_topk
            cache.msa_topk_idx[self.layer_idx] = topk_idx
            cache.last_s_per_doc[self.layer_idx] = s_per_doc

        hidden_states = residual + out
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, s_per_doc


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class MSAQwen3_5Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    routing_scores: Optional[torch.FloatTensor] = None
    per_layer_routing: Optional[dict] = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MSAQwen3_5Model(nn.Module):
    def __init__(self, config: MSAQwen3_5Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        drop = config.msa_dropout
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.layers = nn.ModuleList(
            [MSAQwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.gradient_checkpointing = bool(
            getattr(config, "msa_gradient_checkpointing", False)
        )

    def _set_gradient_checkpointing(self, enable: bool) -> None:
        self.gradient_checkpointing = bool(enable)

    def _build_position_embeddings(self, start: int, end: int, device, ref: torch.Tensor):
        """Return ``(cos, sin)`` suitable for partial-RoPE in BLHD layout.

        ``cos, sin`` shape is ``(B, L, rotary_dim)`` where ``rotary_dim`` equals
        ``head_dim * partial_rotary_factor``; broadcasting to a heads axis is
        done inside :func:`apply_rotary_pos_emb` via ``unsqueeze_dim``.
        """
        position_ids = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0)
        cos, sin = self.rotary_emb(ref, position_ids=position_ids)
        return cos.to(ref.dtype), sin.to(ref.dtype)

    # --------------------- gradient-checkpoint helpers ----------------------
    # These wrappers exist so that ``torch.utils.checkpoint`` can re-execute
    # one decoder layer's worth of work without keeping its activations alive.
    # Returns are pure tensors (placeholders for the no-op branches) because
    # checkpoint with ``use_reentrant=False`` happily handles fixed-shape tuple
    # outputs but not None / dict / nested tuple of tuples.
    def _doc_layer_step(
        self,
        layer: "MSAQwen3_5DecoderLayer",
        h_d: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        doc_pad: Optional[torch.Tensor],
    ):
        pos_emb = (cos, sin)
        if layer.is_full_attention and layer.is_msa_layer:
            K_bar, V_bar, Kr_bar = layer.pool_doc_kvr(h_d, pos_emb, doc_pad)
        else:
            zero = h_d.new_zeros(0)
            K_bar = V_bar = Kr_bar = zero
        h_d_new = layer.forward_doc(h_d, pos_emb, attention_mask=doc_pad)
        return h_d_new, K_bar, V_bar, Kr_bar

    def _query_layer_step(
        self,
        layer: "MSAQwen3_5DecoderLayer",
        h_q: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_pad: Optional[torch.Tensor],
        K_bar: Optional[torch.Tensor],
        V_bar: Optional[torch.Tensor],
        Kr_bar: Optional[torch.Tensor],
        pos_doc_labels: Optional[torch.Tensor],
        aux_tau: float,
    ):
        pos_emb = (cos, sin)
        if (
            layer.is_full_attention
            and layer.is_msa_layer
            and K_bar is not None
        ):
            h_q_new, aux_i, s_per_doc = layer.forward_query_msa(
                h_q,
                pos_emb,
                (K_bar, V_bar, Kr_bar),
                pos_doc_labels,
                aux_tau,
                attention_mask=q_pad,
            )
            return h_q_new, aux_i, s_per_doc
        h_q_new = layer.forward_query_self(h_q, pos_emb, attention_mask=q_pad)
        return h_q_new, h_q_new.new_zeros(()), h_q_new.new_zeros(0)

    @torch.inference_mode()
    def encode_docs(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode-only path used by the offline inference pipeline.

        Runs Stage 1 of the training forward (doc encoding) without going on
        to Stage 2 (query routing). For each MSA layer, returns the pooled
        ``(K_bar, V_bar, Kr_bar)`` tuple. K/V shape is ``(B, N, n_chunks,
        n_kv_heads, head_dim)``; K_R is ``(B, N, n_chunks, n_router_heads,
        head_dim)``.

        Use case: ``msa.inference.offline_encoder`` calls this on every batch
        of corpus docs and persists the per-layer pooled tensors to disk.
        """
        B, N, L_d = doc_input_ids.shape
        device = doc_input_ids.device

        h_d = self.embed_tokens(doc_input_ids.reshape(B * N, L_d))
        doc_pos_emb = self._build_position_embeddings(0, L_d, device, h_d)
        doc_pad = (
            doc_attention_mask.reshape(B * N, L_d).to(h_d.dtype)
            if doc_attention_mask is not None
            else None
        )

        cos_d, sin_d = doc_pos_emb
        pooled_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            h_d, Kb, Vb, Krb = self._doc_layer_step(
                layer, h_d, cos_d, sin_d, doc_pad,
            )
            if layer.is_full_attention and layer.is_msa_layer:
                K_bar = Kb.view(B, N, *Kb.shape[1:])
                V_bar = Vb.view(B, N, *Vb.shape[1:])
                Kr_bar = Krb.view(B, N, *Krb.shape[1:])
                pooled_cache[i] = (K_bar, V_bar, Kr_bar)
        return pooled_cache

    @torch.inference_mode()
    def prefill_query_with_memory(
        self,
        query_input_ids: torch.Tensor,
        pooled_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        query_attention_mask: Optional[torch.Tensor] = None,
        return_last_hidden: bool = True,
    ):
        """Stage 2 inference: encode a query using externally-provided memory.

        Mirrors the second half of :meth:`forward` but takes ``pooled_cache``
        as input (instead of running Stage 1) so the offline encoder's disk
        tensors can be reused across queries.

        Args:
            query_input_ids: ``(B, L_q)``.
            pooled_cache: ``{layer_idx: (K_bar, V_bar, Kr_bar)}`` where each
                tensor has shape ``(B, N_sel, n_chunks, n_heads, head_dim)``
                — i.e. already restricted to the docs the caller wants to
                attend to. For the small-corpus path we just pass the full
                corpus; for large-corpus inference the caller pre-routes
                the top-k docs per layer (see ``sparse_generator``).
            query_attention_mask: optional ``(B, L_q)`` mask.
            return_last_hidden: if True, applies final ``self.norm`` so the
                LM head can be applied directly.

        Returns:
            (hidden_states, per_layer_routing) — hidden_states is
            ``(B, L_q, D)``; per_layer_routing maps MSA layer idx → s_per_doc
            ``(B, N_sel)`` so the caller can rank docs.
        """
        _, L_q = query_input_ids.shape
        device = query_input_ids.device

        h_q = self.embed_tokens(query_input_ids)
        query_offset = self.config.msa_top_k
        q_pos_emb = self._build_position_embeddings(
            query_offset, query_offset + L_q, device, h_q,
        )
        q_pad = (
            query_attention_mask.to(h_q.dtype)
            if query_attention_mask is not None else None
        )

        cos_q, sin_q = q_pos_emb
        per_layer_routing: dict[int, torch.Tensor] = {}
        for i, layer in enumerate(self.layers):
            is_msa_full = (
                layer.is_full_attention
                and layer.is_msa_layer
                and i in pooled_cache
            )
            if is_msa_full:
                Kb, Vb, Krb = pooled_cache[i]
            else:
                Kb = Vb = Krb = None
            h_q, _aux, s_per_doc = self._query_layer_step(
                layer, h_q, cos_q, sin_q, q_pad,
                Kb, Vb, Krb,
                pos_doc_labels=None,
                aux_tau=self.config.msa_aux_tau,
            )
            if is_msa_full:
                per_layer_routing[i] = s_per_doc

        if return_last_hidden:
            h_q = self.norm(h_q)
        return h_q, per_layer_routing

    @torch.inference_mode()
    def forward_query_step(
        self,
        query_input_ids: torch.Tensor,
        pooled_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        kv_cache: "MSAQueryKVCache",
        return_last_hidden: bool = True,
    ):
        """Streaming query forward — used for both prefill and per-step decode.

        First call: pass the entire prompt as ``query_input_ids`` with a
        freshly-constructed ``MSAQueryKVCache``. The router fires on every
        MSA layer, top-k docs are picked + cached, and self-attn KV is
        captured.

        Subsequent calls: pass only the newly-sampled token (``L=1``) plus
        the same ``kv_cache``. RoPE positions are derived from
        ``kv_cache.seq_len`` so each token lands at the right offset, the
        router is bypassed (frozen top-k from prefill is reused), and self
        K/V are appended to the cache.

        Returns ``(hidden_states, per_layer_routing_or_None)``. The routing
        dict is only populated on the prefill step.
        """
        _, L_q = query_input_ids.shape
        device = query_input_ids.device

        h_q = self.embed_tokens(query_input_ids)
        query_offset = self.config.msa_top_k
        start = query_offset + kv_cache.seq_len
        q_pos_emb = self._build_position_embeddings(
            start, start + L_q, device, h_q,
        )
        cache_position = torch.arange(
            kv_cache.seq_len, kv_cache.seq_len + L_q, device=device, dtype=torch.long,
        )

        per_layer_routing: dict[int, torch.Tensor] = {}
        for i, layer in enumerate(self.layers):
            is_msa_full = (
                layer.is_full_attention
                and layer.is_msa_layer
                and i in pooled_cache
            )
            if is_msa_full:
                Kb, Vb, Krb = pooled_cache[i]
                h_q, s_per_doc = layer.forward_query_msa_cached(
                    h_q, q_pos_emb, (Kb, Vb, Krb), kv_cache,
                )
                if s_per_doc is not None:
                    per_layer_routing[i] = s_per_doc
            else:
                h_q = layer.forward_query_self_cached(
                    h_q, q_pos_emb, kv_cache, cache_position=cache_position,
                )

        kv_cache.seq_len += L_q

        if return_last_hidden:
            h_q = self.norm(h_q)
        return h_q, per_layer_routing

    def forward(
        self,
        doc_input_ids: torch.Tensor,
        query_input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        pos_doc_labels: Optional[torch.Tensor] = None,
    ):
        B, N, L_d = doc_input_ids.shape
        _, L_q = query_input_ids.shape
        device = doc_input_ids.device

        # ---------------- Stage 1: encode docs, collecting pooled KVR on MSA layers
        h_d = self.embed_tokens(doc_input_ids.reshape(B * N, L_d))
        h_d = self.dropout(h_d)
        doc_pos_emb = self._build_position_embeddings(0, L_d, device, h_d)
        doc_pad = (
            doc_attention_mask.reshape(B * N, L_d).to(h_d.dtype)
            if doc_attention_mask is not None
            else None
        )

        cos_d, sin_d = doc_pos_emb
        use_ckpt = self.gradient_checkpointing and self.training

        pooled_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            is_msa_full = layer.is_full_attention and layer.is_msa_layer
            if use_ckpt:
                h_d, Kb, Vb, Krb = _ckpt(
                    self._doc_layer_step,
                    layer, h_d, cos_d, sin_d, doc_pad,
                    use_reentrant=False,
                )
            else:
                h_d, Kb, Vb, Krb = self._doc_layer_step(
                    layer, h_d, cos_d, sin_d, doc_pad,
                )
            if is_msa_full:
                K_bar = Kb.view(B, N, *Kb.shape[1:])
                V_bar = Vb.view(B, N, *Vb.shape[1:])
                Kr_bar = Krb.view(B, N, *Krb.shape[1:])
                pooled_cache[i] = (K_bar, V_bar, Kr_bar)

        # ---------------- Stage 2: encode query, doing MSA on MSA layers
        h_q = self.embed_tokens(query_input_ids)
        h_q = self.dropout(h_q)
        query_offset = self.config.msa_top_k
        q_pos_emb = self._build_position_embeddings(
            query_offset, query_offset + L_q, device, h_q
        )
        q_pad = (
            query_attention_mask.to(h_q.dtype)
            if query_attention_mask is not None
            else None
        )

        aux_sum = h_q.new_zeros(())
        aux_count = 0
        per_layer_routing: dict[int, torch.Tensor] = {}
        cos_q, sin_q = q_pos_emb

        for i, layer in enumerate(self.layers):
            is_msa_full = (
                layer.is_full_attention and layer.is_msa_layer and i in pooled_cache
            )
            if is_msa_full:
                Kb, Vb, Krb = pooled_cache[i]
            else:
                Kb = Vb = Krb = None

            if use_ckpt:
                h_q, aux_i, s_per_doc = _ckpt(
                    self._query_layer_step,
                    layer, h_q, cos_q, sin_q, q_pad,
                    Kb, Vb, Krb, pos_doc_labels, self.config.msa_aux_tau,
                    use_reentrant=False,
                )
            else:
                h_q, aux_i, s_per_doc = self._query_layer_step(
                    layer, h_q, cos_q, sin_q, q_pad,
                    Kb, Vb, Krb, pos_doc_labels, self.config.msa_aux_tau,
                )

            if is_msa_full:
                aux_sum = aux_sum + aux_i
                aux_count += 1
                per_layer_routing[i] = s_per_doc

        h_q = self.norm(h_q)
        aux_loss = aux_sum / max(aux_count, 1)
        last_routing = (
            per_layer_routing[max(per_layer_routing)] if per_layer_routing else None
        )
        return h_q, aux_loss, last_routing, per_layer_routing


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------
class MSAQwen3_5ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MSAQwen3_5Config

    def __init__(self, config: MSAQwen3_5Config):
        super().__init__(config)
        self.model = MSAQwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    # ---------------------------------------------------- gradient checkpointing
    # We bypass HF's default ``_set_gradient_checkpointing`` machinery because
    # MSA's two-pass forward (doc encode -> query) doesn't fit the layer-list
    # iteration HF expects. Instead we expose tiny enable/disable hooks that
    # toggle the flag the model's forward checks for itself.
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model._set_gradient_checkpointing(True)

    def gradient_checkpointing_disable(self):
        self.model._set_gradient_checkpointing(False)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.model.gradient_checkpointing

    # ------------------------------------------------------- weight loading
    def load_qwen3_5_pretrained(
        self, qwen3_5_path: str, dtype: torch.dtype = torch.bfloat16
    ) -> dict:
        """Load the backbone from a Qwen3.5-9B-Base HF checkpoint (text-only).

        Router projectors (``kr_proj`` / ``qr_proj``) keep their fresh init.
        Vision weights and MTP weights are dropped. Returns a small report.
        """
        from transformers import Qwen3_5ForCausalLM

        src = Qwen3_5ForCausalLM.from_pretrained(qwen3_5_path, dtype=dtype)
        src_sd = src.state_dict()

        own = self.state_dict()
        loaded, skipped_from_src, missing_in_src = [], [], []
        for k, v in src_sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded.append(k)
            else:
                skipped_from_src.append(k)
        for k in own.keys():
            if k not in src_sd:
                missing_in_src.append(k)
        self.load_state_dict(own, strict=False)
        del src, src_sd
        return {
            "loaded_count": len(loaded),
            "skipped_from_src_count": len(skipped_from_src),
            "missing_in_src_count": len(missing_in_src),
            "missing_in_src_sample": missing_in_src[:10],
            "skipped_from_src_sample": skipped_from_src[:10],
        }

    # ------------------------------------------------------------- forward
    def forward(
        self,
        doc_input_ids: torch.Tensor,
        query_input_ids: torch.Tensor,
        doc_attention_mask: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        pos_doc_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        lm_loss_coef: float = 1.0,
        aux_loss_coef: Optional[float] = None,
        logits_to_keep: int = 0,
        **_unused,
    ):
        hidden_states, aux_loss, routing, per_layer = self.model(
            doc_input_ids=doc_input_ids,
            query_input_ids=query_input_ids,
            doc_attention_mask=doc_attention_mask,
            query_attention_mask=query_attention_mask,
            pos_doc_labels=pos_doc_labels,
        )
        slice_idx = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else slice(None)
        )
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        lm_loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100
            )

        total = None
        if lm_loss is not None:
            ac = aux_loss_coef if aux_loss_coef is not None else self.config.msa_aux_coef
            total = lm_loss_coef * lm_loss + ac * aux_loss

        return MSAQwen3_5Output(
            loss=total,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            logits=logits,
            routing_scores=routing,
            per_layer_routing=per_layer,
        )
