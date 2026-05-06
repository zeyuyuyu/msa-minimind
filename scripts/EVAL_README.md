# MSA Eval Pipeline

End-to-end evaluation suite for **MSA-Qwen3.5-9B** checkpoints, plus a
**vanilla-Instruct baseline** for backbone-comparison experiments. Implements
the paper-aligned 10-benchmark LLM-judge pipeline used to produce
`msa_qwen3_5_9b_eval_report.pdf` (Tables 2/3/Fig 1) and the RULER NIAH
heatmap (Fig 4).

## TL;DR — three-script flow

```
                         per-bench JSON                 *_llmscore.json
ckpt + bench_root  ───►  bench_evermind_aligned_v2  ───► llm_judge_evermind ──► summary
                              (script 1)                    (script 2)
```

```bash
# 1) MSA model on 10 benches, 100 queries / bench, top_k=10 retrieved docs
bash scripts/run_bench_on_ckpt.sh \
    /workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt \
    sft_s2_full 100 \
    "dureader,2wikimultihopqa,hipporag_narrative,hipporag_popqa,triviaqa_06M,triviaqa_10M,hotpotqa,musique,nature_questions,msmarco_v1"

# 2) LLM judge (gemini-2.5-flash via OpenRouter, paper-aligned 0-5 rubric)
export OPENROUTER_API_KEY=sk-or-v1-...
bash scripts/run_llm_judge.sh /workspace/runs/eval_p5/sft_s2_full
```

Output: `<eval_dir>/llmscore_summary.json`, `<eval_dir>/<bench>_llmscore.json`.

## Vanilla-Instruct baseline (NEW)

To isolate **backbone (base vs instruct) effects from the MSA pipeline**, this
branch adds `scripts/bench_vanilla_instruct.py`: a self-contained script that
runs vanilla `Qwen3.5-9B-Instruct` (no MSA, no LoRA) on the **same 10 benches
and same record schema** so the same `llm_judge_evermind.py` can score it.

Three eval modes:

| `--mode`  | Context fed to the LM            | Tests                              |
|-----------|----------------------------------|------------------------------------|
| `oracle`  | gold reference docs verbatim     | LM upper-bound reading-comp        |
| `bm25`    | BM25 top-K docs (built in)       | paper Table 2 R@K analog           |
| `noctx`   | nothing (closed-book)            | LM intrinsic QA knowledge          |

```bash
# Run-A: oracle (LM upper bound)
python3 scripts/bench_vanilla_instruct.py \
    --qwen_path /workspace/qwen35_instruct \
    --bench_root /workspace/msa_bench \
    --out_dir /workspace/eval_vanilla_instruct/oracle_full \
    --mode oracle --num_queries 100 --max_new_tokens 1024 --enable_thinking 0

# Run-B: BM25 retrieval (paper-aligned RAG)
python3 scripts/bench_vanilla_instruct.py \
    --out_dir /workspace/eval_vanilla_instruct/bm25_top5 \
    --mode bm25 --top_k 5 --num_queries 100

# Run-C: closed-book
python3 scripts/bench_vanilla_instruct.py \
    --out_dir /workspace/eval_vanilla_instruct/noctx \
    --mode noctx --num_queries 100

# Then the same judge:
bash scripts/run_llm_judge.sh /workspace/eval_vanilla_instruct/oracle_full
```

The output JSON schema (`record_list[*].true_answer / pred_answer`) is identical
to `bench_evermind_aligned_v2.py`, so the existing judge picks it up unchanged.

## Output schema

Each `<bench>.json` written by either runner contains:

```json
{
  "anonymous": {
    "precision": {
      "metrics": {"precision": 0.05, "recall": 0.42, "f1": 0.09, "iou": 0.05},
      "record_list": [
        {
          "labels_id": [12, 47],
          "pred_id":   [47, 88, 3],
          "question":  "...",
          "true_answer": "...",
          "pred_answer": "...",
          "generated_text": "<full LM output incl. thinking/format tokens>",
          "predict_context": [{"0": "doc text"}],
          "gt_context":      [{"0": "doc text"}]
        }
      ]
    }
  },
  "telemetry": {
    "empty_answer_rate": 0.91,
    "reached_part_c_rate": 1.0,
    "reached_im_end_rate": 1.0,
    "avg_n_chars": 569,
    "median_n_chars": 555,
    "n_obj_ref_end_dist": {"2": 94, "3": 4, "4": 2}
  },
  "config": {"max_new_tokens": 1024, "top_k": 10}
}
```

## RULER NIAH (Fig 4 of report)

For the long-context heatmap (32K → 1M, 8 NIAH sub-tasks):

```bash
python3 scripts/niah_ruler_runner.py \
    --qwen3_5_path /workspace/qwen35_base \
    --ckpt /workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt \
    --out_dir /workspace/logs/ruler_pg \
    --context_lengths 32000,64000,128000,256000,512000,1000000 \
    --n_trials 10 --top_k 16
```

## Bench dataset prep

`bench_root` is expected to contain one subdirectory per bench, each with two
serialized files (`qdata_*` queries, `mdata_*` corpus docs):

```
/workspace/msa_bench/
├── msmarco_v1/
│   ├── qdata_msmarco_v1.pkl    # list of {query, answer, reference_list}
│   └── mdata_msmarco_v1.pkl    # list of doc strings (positional id == index)
├── nature_questions/
│   ├── qdata_nature_questions.pkl
│   └── mdata_nature_questions.pkl
└── ... (10 benches total)
```

For MSA inference (`bench_evermind_aligned_v2.py`) you also need pre-encoded
sparse-key / value tensors for each corpus in
`/workspace/encoded_corpora/<bench>_<size>/` (built once via
`scripts/encode_*` helpers in this directory).

For the vanilla-instruct baseline, **only** the two corpus files are needed —
no pre-encoded sparse corpus, no MSA model.

## Dependencies

- transformers >= 4.45 (Qwen3.5 support, multimodal arch)
- torch with CUDA (>= 2.4)
- numpy, tqdm
- For LLM judge: requests + valid `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`)

No external retrieval library needed — `bench_vanilla_instruct.py` ships a
self-contained numpy BM25.

## File map (this branch)

```
scripts/
├── bench_evermind_aligned_v2.py     # main MSA-9B eval (10 benches, 100q/each)
├── bench_evermind_aligned.py        # v1 kept for reference
├── bench_runner.py                  # MSA model loader + LoRA wrapper
├── bench_vanilla_instruct.py        # NEW vanilla-Instruct baseline (3 modes)
├── llm_judge_evermind.py            # paper-aligned 0-5 LLM judge
├── niah_runner.py                   # NIAH single-corpus runner (Fig 1 anchor)
├── niah_ruler_runner.py             # RULER 8-task NIAH runner (Fig 4 heatmap)
├── run_bench_on_ckpt.sh             # MSA eval wrapper
├── run_llm_judge.sh                 # judge wrapper
└── EVAL_README.md                   # this file
msa/
├── model_msa_qwen3_5.py             # MSA-Qwen3.5 architecture
├── lora_wrap.py                     # LoRA injection
└── inference/                       # SparseGenerator + offline encoder
```
