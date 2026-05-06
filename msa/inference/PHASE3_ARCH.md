# Phase 3 Inference Engine Architecture

## Goal

Build a three-stage MSA inference engine for **Qwen3.5-9B-MSA-SFT** that
matches the protocol used by the EverMind-AI/MSA reference benchmark
(`src/msa_service.py` + `src/app/benchmark.py`), so we can run

    scripts/run_benchmarks.sh

against our trained checkpoint and produce paper-aligned metrics on:

| Tier | Benchmarks | Context | Phase 3 milestone |
|---|---|---|---|
| Short | msmarco_v1 / hotpotqa / nq / musique / 2wiki | ≤32K | W1 — first EM/F1 |
| Medium | triviaqa_06M | 0.6M | W3 |
| Long | hipporag_popqa, hipporag_narrative | ~1M | W4 |
| **Headline** | **ms_100M** | **100M (961K docs)** | **W6-7 (Phase 4)** |

## Why our model can't reuse EverMind's engine verbatim

EverMind's MSA-4B uses **pure-dense Qwen3** (every layer is `Qwen3Attention`).
Their `prefill.py` / `msa_service.py` / `MemorySparseAttention.forward`
assume every layer has a per-token KV cache.

Our **Qwen3.5-9B-MSA** is **hybrid**: 8 of 32 layers are `Qwen3_5GatedAttention`
(MSA), 24 are `Qwen3_5GatedDeltaNet` (linear attention with recurrent state).
The 24 GDN layers have **no KV cache** — they have a per-doc recurrent state
during encoding and a separate query-only state during decode.

So the EverMind cache machinery and multi-process worker design need an
**adapter layer** before they apply. The router-engine + sparse-generator
half is mostly portable; the prefill half is not.

## Three-stage protocol (our adaptation)

### Stage 0 — Tokenization & template

Per the EverMind prompt protocol (`src/utils/tools.py::compose_input` +
`src/msa_service.py:1725`):

    docs:    "<|im_start|>[i]. {doc}[i]<|im_end|>"  per doc
    query:   "\nPlease answer the question based on the above historical
              document information\n\n{question}\nPlease return all
              documents related to the question\n"

We strip per-doc `[i]` markers from the query stream's `doc_ids` mask
(the markers themselves are NOT memory tokens; they're separator tokens).

Implemented in `prompt_template.py`. ✅

### Stage 1 — Offline doc encoding

Run our trained MSAQwen3_5 model forward in **training mode** over each
batch of docs:

* All 32 layers run, including the 24 GatedDeltaNet layers.
* GDN layers process each doc as an independent batch item, so each doc
  has its own recurrent state.
* On each of the 8 MSA (full-attention) layers, we extract the chunk-pooled
  tensors:

      K_chunk[layer]  shape (n_kv_heads, n_chunks, head_dim)  — content K
      V_chunk[layer]  shape (n_kv_heads, n_chunks, head_dim)  — content V
      KR_chunk[layer] shape (n_router_heads, n_chunks, head_dim) — router K
      doc_id_per_chunk[layer]  shape (n_chunks,)              — origin doc

* Persist to disk per benchmark, partitioned by `doc_idx_range`. Format:
  one `{benchmark}_{layer}.pt` file per MSA layer with the four tensors
  concatenated, plus a `meta.json` with corpus size + chunk_size.

Implemented in `offline_encoder.py`. 🚧 In-progress.

### Stage 2 — Online routing

Per-query, given the loaded memory K_R bank:

* Tokenize template prefix + query, run through model up to the first
  MSA layer.
* For each MSA layer:
    1. Compute Q_R for the query tokens (`qr_proj`).
    2. Score against `KR_chunk[layer]` (dot product, optional norm),
       reduce-over-heads, take top_k chunks.
    3. Build a sparse attention pattern: query attends to (template_KV +
       selected chunks' content KV from `K_chunk[layer]` / `V_chunk[layer]`).
* GDN layers continue with query-only recurrent state (no docs in their
  context — important: this is a deviation from MSA-4B but matches how
  Qwen3.5-9B is actually trained).

Implemented in `router_engine.py`. 🚧

### Stage 3 — Sparse generation

Token-by-token generation continuing the sparse pattern:

* MSA layers: append generated-token KV to a per-step query KV cache;
  attention sees template + selected memory + query history.
* GDN layers: append to recurrent state per generated token.
* Stop on EOS or `max_generate_tokens`.
* Output: generated text → `parse_response` → `(retrieved_doc_ids, answer)`.

Implemented in `sparse_generator.py`. 🚧

### Stage 4 — Compression for long context (Phase 4)

* **4-bit KV quantization** on `K_chunk` / `V_chunk`. Router K stays bf16
  on GPU (it's small, ~10MB per layer for 100M corpus).
* **Tiered storage**:
  - GPU: KR_chunk for all 8 MSA layers (~80MB at 100M corpus)
  - CPU DRAM: K_chunk + V_chunk for hot chunks (~50GB after 4-bit quant)
  - NVMe: cold chunks
* **Async prefetch**: while generating token i, prefetch chunks for token
  i+1's likely top-k.

Implemented in `kv_quantizer.py` + `disk_layout.py`. ⏳ Phase 4.

## Concrete numbers for ms_100M corpus

Confirmed via `inspect_pkl.py` on the downloaded data:

* 961,686 documents, ~113.8M characters → **~28M words, ~38M tokens at
  modern Qwen tokenization** (paper marketing says "100M tokens" assuming
  more aggressive tokenization, but in practice it's ~30-40M token ranges
  that we route over).

Per-MSA-layer storage (chunk_size=64, 8 KV heads, head_dim=256):

* K_chunk: 38M tok / 64 = 594K chunks × 8 × 256 × 2 bytes = **2.4 GB / layer**
* V_chunk: same = **2.4 GB / layer**
* KR_chunk (8 router heads): same = **2.4 GB / layer**

For 8 MSA layers:
* Content K+V: 8 × 2 × 2.4 = **38.4 GB** (4-bit → 9.6 GB after quant)
* Router K: 8 × 2.4 = **19.2 GB** (kept bf16 on GPU; needs > 1× H200's 143GB
  budget, so we'll page router K across the 8 MSA layers too in Phase 4)

H200 single-card budget for paper-100M demo:
* Backbone + activations: ~30 GB
* Router K (paged): ~5 GB at any time
* Working content KV (top-k chunks): ~1 GB
* Total: ~36 GB → comfortable headroom for generation.

## File map

    msa/inference/
      __init__.py              ✅ shipped
      memory_loader.py         ✅ shipped + smoke-tested
      prompt_template.py       ✅ shipped + smoke-tested
      offline_encoder.py       🚧 next
      router_engine.py         🚧 W2
      sparse_generator.py      🚧 W2
      kv_quantizer.py          ⏳ Phase 4
      disk_layout.py           ⏳ Phase 4
      bench_runner.py          🚧 wires the above + EM/F1
      PHASE3_ARCH.md           this doc
