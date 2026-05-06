"""Online sparse generator for MSA-Qwen3.5-9B inference.

This is the §3.2 stage-2 + stage-3 inference pipeline (router → sparse attend
→ generate) glued onto a pre-trained MSAQwen3_5ForCausalLM.

For a single query, the protocol is:

  1. Tokenise the query (optionally wrapped with the EverMind prompt).
  2. (Pre-route, two-pass mode only) Run a *probe* pass through the model
     using the **full** corpus pooled_cache to capture per-layer ``s_per_doc``;
     pick top-k docs per MSA layer.
  3. (Final pass) Re-run the query through the model, but this time on every
     MSA layer the pooled_cache is restricted to that layer's top-k docs. This
     mirrors what the §3.2 inference path does in the EverMind reference.
  4. (Optional) Autoregressively decode tokens off the final hidden state.

For now we ship the **single-pass** path, which is mathematically equivalent
to step 3 when the corpus is small enough that all docs participate. Top-k
restriction + multi-pass is a TODO once we move past 64-doc smoke tests.

Generation (step 4) is left as a follow-up; the immediate utility of this
file is to power retrieval-only benchmarks (precision/recall/F1/IoU on
predicted doc IDs) which is what the `bench_runner` will call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from msa.model_msa_qwen3_5 import (
    MSAQwen3_5ForCausalLM,
    MSAQueryKVCache,
)
from msa.inference.router_engine import EncodedCorpus


def _sample_from_logits(
    logits: torch.Tensor, temperature: float, top_p: float
) -> int:
    """Pick one token id from a 1-D logits tensor with temperature/top-p."""
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    if 0 < top_p < 1:
        sorted_p, sorted_idx = probs.sort(descending=True)
        cum = sorted_p.cumsum(dim=-1)
        cutoff = (cum > top_p).nonzero()
        n_keep = int(cutoff[0]) + 1 if len(cutoff) > 0 else len(probs)
        keep_p = sorted_p[:n_keep]
        keep_p = keep_p / keep_p.sum()
        pick = int(torch.multinomial(keep_p, 1).item())
        return int(sorted_idx[pick].item())
    return int(torch.multinomial(probs, 1).item())


@dataclass
class RetrievalResult:
    """Output of ``SparseGenerator.retrieve(...)`` for one query.

    All ``(B, ...)`` tensors below have B=1 in the smoke path.
    """

    pred_doc_ids: list[int]                                # length k_docs
    per_layer_doc_ids: dict[int, list[int]] = field(default_factory=dict)
    last_layer_scores: Optional[torch.Tensor] = None       # (n_docs,) raw s_per_doc
    elapsed_sec: float = 0.0


class SparseGenerator:
    """Single-query MSA inference engine over an offline-encoded corpus.

    Construct once per ``(model, encoded_corpus)`` pair, then call
    ``retrieve`` (or eventually ``generate``) repeatedly for each new
    query — the encoded corpus is kept on the chosen device and reused
    without re-running the offline encoder.
    """

    def __init__(
        self,
        model: MSAQwen3_5ForCausalLM,
        tokenizer,
        encoded_corpus: dict[int, EncodedCorpus],
        device: str | torch.device = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        # Move corpus tensors onto the model's device once.
        self.corpus: dict[int, EncodedCorpus] = {
            i: c.to(device=self.device) for i, c in encoded_corpus.items()
        }
        any_layer = next(iter(self.corpus.values()))
        self.n_docs: int = int(any_layer.n_docs)
        self.doc_ids: torch.Tensor = any_layer.doc_ids.to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_full_pooled_cache(self) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Reshape every layer's doc-grouped tensors to the (B=1, N, ...) shape
        that ``model.prefill_query_with_memory`` expects."""
        out: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for layer_idx, c in self.corpus.items():
            out[layer_idx] = (
                c.K_doc.unsqueeze(0),
                c.V_doc.unsqueeze(0),
                c.KR_doc.unsqueeze(0),
            )
        return out

    def _build_topk_pooled_cache(
        self,
        topk_docs_per_layer: dict[int, torch.Tensor],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Per-layer top-k restricted pooled_cache (shape (1, k, n_chunks, …))."""
        out = {}
        for layer_idx, doc_idx_1xK in topk_docs_per_layer.items():
            K, V, KR = self.corpus[layer_idx].gather_topk_docs(doc_idx_1xK)
            out[layer_idx] = (K, V, KR)
        return out

    @staticmethod
    def _aggregate_routing(
        per_layer_routing: dict[int, torch.Tensor],
        agg: str = "last",
    ) -> torch.Tensor:
        """Reduce per-MSA-layer s_per_doc into one (B, n_docs) score vector.

        agg modes:
          * 'last' — only the deepest MSA layer (paper-aligned final routing)
          * 'mean' — average across MSA layers (more robust for evaluation)
          * 'max'  — max across layers
        """
        if not per_layer_routing:
            raise ValueError("empty per_layer_routing — was the model run?")
        if agg == "last":
            last_idx = max(per_layer_routing.keys())
            return per_layer_routing[last_idx]
        stacked = torch.stack(list(per_layer_routing.values()), dim=0)  # (L, B, N)
        if agg == "mean":
            return stacked.mean(dim=0)
        if agg == "max":
            return stacked.amax(dim=0)
        raise ValueError(f"unknown agg={agg!r}")

    # ------------------------------------------------------------------
    # Public API: single-query retrieval
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def retrieve(
        self,
        query_text: str,
        k_docs: int = 8,
        max_query_len: int = 1024,
        agg: str = "last",
        use_topk_restriction: bool = False,
        topk_per_layer: int = 64,
    ) -> RetrievalResult:
        """Score ``query_text`` against the loaded corpus and return top-k doc ids.

        Args:
            query_text: raw query string (the caller is responsible for any
                EverMind-style instruction wrapping; for retrieval-only the
                wrap is optional but recommended for parity with training).
            k_docs: number of doc indices to return at the end.
            max_query_len: cap query token count to keep prefill bounded.
            agg: how to reduce per-MSA-layer s_per_doc — 'last' / 'mean' / 'max'.
            use_topk_restriction: if True, do a two-pass pre-route then
                attend only to top-``topk_per_layer`` docs per layer. Mirrors
                the §3.2 inference path; slower but matches paper. For small
                corpora it's a no-op since we'd select the entire corpus.
            topk_per_layer: how many docs to keep per MSA layer in pass 2.
        """
        import time
        t0 = time.time()
        ids = self.tokenizer.encode(query_text, add_special_tokens=False)[:max_query_len]
        q_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        # ---- Pass 1: full-corpus stage-2 forward -> per-layer routing ----
        full_cache = self._build_full_pooled_cache()
        _, per_layer_routing = self.model.model.prefill_query_with_memory(
            query_input_ids=q_ids,
            pooled_cache=full_cache,
            return_last_hidden=False,
        )

        # ---- Optional pass 2: restrict to per-layer top-k and re-run ----
        if use_topk_restriction and topk_per_layer < self.n_docs:
            topk_docs_per_layer: dict[int, torch.Tensor] = {}
            for layer_idx, s_per_doc in per_layer_routing.items():
                k = min(topk_per_layer, s_per_doc.shape[-1])
                _, idx = s_per_doc.topk(k, dim=-1)
                topk_docs_per_layer[layer_idx] = idx
            restricted_cache = self._build_topk_pooled_cache(topk_docs_per_layer)
            _, per_layer_routing = self.model.model.prefill_query_with_memory(
                query_input_ids=q_ids,
                pooled_cache=restricted_cache,
                return_last_hidden=False,
            )
            # Translate back from "row in topk slice" to "global doc index".
            translated: dict[int, torch.Tensor] = {}
            for layer_idx, s in per_layer_routing.items():
                # s shape: (1, k); doc_idx_1xK gives global doc rows
                global_idx = topk_docs_per_layer[layer_idx]                    # (1, k)
                # We rebuild s_per_doc back over the full corpus axis so the
                # downstream aggregator stays uniform; missing docs get -inf.
                full = s.new_full((1, self.n_docs), float("-inf"))
                full.scatter_(1, global_idx, s)
                translated[layer_idx] = full
            per_layer_routing = translated

        # ---- Aggregate to a single score vector and pick top-k docs ----
        scores = self._aggregate_routing(per_layer_routing, agg=agg)        # (1, n_docs)
        k = min(k_docs, scores.shape[-1])
        _, top_idx = scores.topk(k, dim=-1)
        top_global_doc_ids = self.doc_ids[top_idx[0]].tolist()

        # Per-layer top-k for diagnostics
        per_layer_doc_ids: dict[int, list[int]] = {}
        for layer_idx, s in per_layer_routing.items():
            _, idx = s.topk(k, dim=-1)
            per_layer_doc_ids[layer_idx] = self.doc_ids[idx[0]].tolist()

        return RetrievalResult(
            pred_doc_ids=top_global_doc_ids,
            per_layer_doc_ids=per_layer_doc_ids,
            last_layer_scores=scores[0].detach().cpu(),
            elapsed_sec=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Public API: autoregressive generation
    # ------------------------------------------------------------------
    def _routed_pooled_cache(
        self,
        query_input_ids: torch.Tensor,
        topk_per_layer: int,
    ) -> dict:
        """Two-pass route: pass-1 finds top-k docs per MSA layer using the
        full corpus, pass-2 (the caller's actual prefill) restricts attention
        to those top-k docs only.

        Returns a per-layer ``{layer_idx: (K, V, KR)}`` cache where each
        tensor has shape ``(1, k, n_chunks_per_doc, n_heads, head_dim)``.
        This matches what ``forward_query_msa`` expects, so the caller can
        plug the result straight into ``prefill_query_with_memory``.
        """
        # Pass 1: score against the full corpus to find top-k per MSA layer.
        full_cache = self._build_full_pooled_cache()
        _, per_layer = self.model.model.prefill_query_with_memory(
            query_input_ids=query_input_ids,
            pooled_cache=full_cache,
            return_last_hidden=False,
        )
        topk_docs = {}
        for layer_idx, s_per_doc in per_layer.items():
            k = min(topk_per_layer, s_per_doc.shape[-1])
            _, idx = s_per_doc.topk(k, dim=-1)
            topk_docs[layer_idx] = idx
        return self._build_topk_pooled_cache(topk_docs)

    @torch.inference_mode()
    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        eos_token_ids: Optional[Sequence[int]] = None,
        stop_strings: Optional[Sequence[str]] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        return_text: bool = True,
        topk_docs_per_layer: Optional[int] = None,
        reroute_every_n_steps: int = 0,
        use_kv_cache: bool = True,
    ) -> dict:
        """Autoregressive decode using offline MSA memory.

        With ``use_kv_cache=True`` (default), the path is:

        1. Tokenise the prompt; if ``topk_docs_per_layer`` is set, a probe
           pass over the **full** corpus picks the top-``topk_docs_per_layer``
           docs per MSA layer, restricting the rest of decoding to that
           subset.
        2. ``forward_query_step`` runs the full prompt once as **prefill**:
           the router fires on every MSA layer, top-k chunks are gathered
           and frozen on the cache, self-attn ``K``/``V`` are captured per
           full-attention layer, and the GatedDeltaNet conv/recurrent
           states are advanced.
        3. Each subsequent decode step passes a single new token through
           ``forward_query_step``: the router is bypassed (frozen top-k
           reused), self-attn appends one row to the cached ``K``/``V``,
           and the linear layers advance their recurrent state by one
           step. End-to-end this is O(L) per step instead of O(L²).

        With ``use_kv_cache=False``, falls back to the original slow path
        that re-runs prefill from scratch on every step (kept around for
        sanity checks against the cached path).

        Args:
            prompt_text: full chat prompt (use
                :func:`msa.inference.prompt_template.build_msa_train_prompt`
                for the SFT-trained format).
            max_new_tokens: cap on generated tokens.
            eos_token_ids: list of token ids that terminate generation;
                defaults to ``[tokenizer.eos_token_id]`` plus the id of
                ``<|im_end|>`` if available.
            stop_strings: optional list of substrings — if any appears in
                the decoded text-so-far, generation halts.
            temperature: 0.0 = greedy argmax; >0 enables sampling.
            top_p: nucleus sampling cutoff (only used if temperature > 0).

        Returns:
            ``{"text": str, "tokens": list[int], "stopped_by": str,
              "elapsed_sec": float, "n_new_tokens": int,
              "per_layer_routing_last_step": dict[int, list[float]]}``.
        """
        import time
        t0 = time.time()
        if eos_token_ids is None:
            eos_token_ids = []
            if self.tokenizer.eos_token_id is not None:
                eos_token_ids.append(int(self.tokenizer.eos_token_id))
            try:
                im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                if isinstance(im_end, int) and im_end >= 0 and im_end != self.tokenizer.unk_token_id:
                    eos_token_ids.append(int(im_end))
            except Exception:
                pass
        eos_set = set(int(x) for x in eos_token_ids)

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        ids: list[int] = list(prompt_ids)
        prompt_len = len(prompt_ids)

        use_topk = (
            topk_docs_per_layer is not None
            and topk_docs_per_layer > 0
            and topk_docs_per_layer < self.n_docs
        )
        if use_topk:
            q_t0 = torch.tensor([ids], dtype=torch.long, device=self.device)
            active_cache = self._routed_pooled_cache(q_t0, int(topk_docs_per_layer))
            route_phase = "topk"
        else:
            active_cache = self._build_full_pooled_cache()
            route_phase = "full"

        last_routing: dict[int, list[float]] = {}
        stopped_by = "max_new_tokens"

        if use_kv_cache:
            kv_cache = MSAQueryKVCache(self.model.config)
            prompt_t = torch.tensor([ids], dtype=torch.long, device=self.device)
            hidden, per_layer = self.model.model.forward_query_step(
                query_input_ids=prompt_t,
                pooled_cache=active_cache,
                kv_cache=kv_cache,
                return_last_hidden=True,
            )
            logits = self.model.lm_head(hidden[:, -1:, :])
            if temperature <= 0:
                next_id = int(logits.argmax(dim=-1).item())
            else:
                next_id = _sample_from_logits(logits[0, 0], temperature, top_p)
            ids.append(next_id)
            last_routing = {
                lid: per_layer[lid][0].detach().float().cpu().tolist()
                for lid in per_layer
            }

            if next_id in eos_set:
                stopped_by = f"eos_{next_id}"
            else:
                for step in range(1, max_new_tokens):
                    new_t = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                    hidden, _ = self.model.model.forward_query_step(
                        query_input_ids=new_t,
                        pooled_cache=active_cache,
                        kv_cache=kv_cache,
                        return_last_hidden=True,
                    )
                    logits = self.model.lm_head(hidden[:, -1:, :])
                    if temperature <= 0:
                        next_id = int(logits.argmax(dim=-1).item())
                    else:
                        next_id = _sample_from_logits(logits[0, 0], temperature, top_p)
                    ids.append(next_id)

                    if next_id in eos_set:
                        stopped_by = f"eos_{next_id}"
                        break
                    if stop_strings:
                        new_text = self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=False)
                        hit = next((s for s in stop_strings if s in new_text), None)
                        if hit is not None:
                            stopped_by = f"stop_string:{hit}"
                            break

            n_new = len(ids) - prompt_len
            new_text = (
                self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=False)
                if return_text else ""
            )
            return {
                "text": new_text,
                "tokens": ids[prompt_len:],
                "stopped_by": stopped_by,
                "elapsed_sec": time.time() - t0,
                "n_new_tokens": n_new,
                "per_layer_routing_last_step": last_routing,
                "route_phase": route_phase,
                "topk_docs_per_layer": int(topk_docs_per_layer) if use_topk else None,
                "kv_cache": True,
            }

        # ---------------------- legacy slow path (no KV cache) ---------------
        for step in range(max_new_tokens):
            if (
                use_topk
                and reroute_every_n_steps > 0
                and step > 0
                and step % reroute_every_n_steps == 0
            ):
                q_now = torch.tensor([ids], dtype=torch.long, device=self.device)
                active_cache = self._routed_pooled_cache(q_now, int(topk_docs_per_layer))
            q_t = torch.tensor([ids], dtype=torch.long, device=self.device)
            hidden, per_layer = self.model.model.prefill_query_with_memory(
                query_input_ids=q_t,
                pooled_cache=active_cache,
                return_last_hidden=True,
            )
            logits = self.model.lm_head(hidden[:, -1:, :])     # (1, 1, V)
            if temperature <= 0:
                next_id = int(logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)[0, 0]
                if 0 < top_p < 1:
                    sorted_p, sorted_idx = probs.sort(descending=True)
                    cum = sorted_p.cumsum(dim=-1)
                    cutoff = (cum > top_p).nonzero()
                    n_keep = int(cutoff[0]) + 1 if len(cutoff) > 0 else len(probs)
                    keep_p = sorted_p[:n_keep]
                    keep_p = keep_p / keep_p.sum()
                    pick = int(torch.multinomial(keep_p, 1).item())
                    next_id = int(sorted_idx[pick].item())
                else:
                    next_id = int(torch.multinomial(probs, 1).item())
            ids.append(next_id)
            last_routing = {
                lid: per_layer[lid][0].detach().float().cpu().tolist()
                for lid in per_layer
            }

            if next_id in eos_set:
                stopped_by = f"eos_{next_id}"
                break
            if stop_strings:
                # Decode just the new portion for cheap substring check
                new_text = self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=False)
                hit = next((s for s in stop_strings if s in new_text), None)
                if hit is not None:
                    stopped_by = f"stop_string:{hit}"
                    break

        n_new = len(ids) - prompt_len
        new_text = (
            self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=False)
            if return_text else ""
        )
        return {
            "text": new_text,
            "tokens": ids[prompt_len:],
            "stopped_by": stopped_by,
            "elapsed_sec": time.time() - t0,
            "n_new_tokens": n_new,
            "per_layer_routing_last_step": last_routing,
            "route_phase": route_phase,
            "topk_docs_per_layer": int(topk_docs_per_layer) if use_topk else None,
        }
