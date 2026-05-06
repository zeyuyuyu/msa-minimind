"""LLM-judge runner for EverMind-aligned evaluation outputs.

Reads JSON files written by bench_evermind_aligned.py, calls an LLM judge
to score each (question, true_answer, pred_answer) tuple on 0-5 scale,
and writes per-bench *_llmscore.json plus a summary.

Aligned with EverMind src/evaluation/llm_judge.py prompt + parsing.

Backends:
  --backend openrouter  (default)  needs OPENROUTER_API_KEY
                                   uses google/gemini-2.5-flash (paper-aligned)
  --backend openai                 needs OPENAI_API_KEY
                                   uses gpt-4o-mini by default

Run:
  OPENROUTER_API_KEY=... python scripts/llm_judge_evermind.py \
      --in_dir /workspace/eval_evermind_aligned \
      --backend openrouter
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm


def build_score_prompt(gold_answer, model_answer, query):
    return f""""Based on the accuracy, completeness, and relevance of the predicted answer to the real answer in the context of the **query**, assign an objective score from 0 to 5 (5 being the highest, 0 the lowest).

    The scoring must strictly adhere to the following criteria. The final output can only be a single number.

    Scoring Criteria:

    5: The predicted answer is exactly the same as the real answer and correctly answers the query. Differences in wording do not affect factual accuracy.

    4: The predicted answer contains all the core information of the real answer, with no errors, but includes a small amount of non-critical redundant content.

    3: The predicted answer captures the core information but differs from the real answer in some aspects. The predicted answer is slightly incomplete or imprecise, but contains no errors.

    2: The predicted answer is partially relevant to the real answer but omits a significant amount of information or deviates from the core topic of the query.

    1: The predicted answer attempts to address the query (maintains basic relevance to the topic) but provides factually incorrect information. It does not contradict the core claim of the real answer, but shows incomplete or inaccurate understanding of the topic.

    0. The predicted answer is completely unrelated to the query, consists of gibberish, or is a pure hallucination that shares no logical connection with the real answer.

    Query:

    {query}

    True Answer:

    {gold_answer}

    Predicted Answer:

    {model_answer}

    Output only a single number (0, 1, 2, 3, 4, or 5): """


def parse_score(text):
    text = (text or "").strip()
    for ch in text:
        if ch.isdigit() and int(ch) <= 5:
            return int(ch)
    return 0


def make_client_and_model(backend: str, model_override: str | None):
    from openai import OpenAI
    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY not set")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        model = model_override or "google/gemini-2.5-flash"
    elif backend == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set")
        client = OpenAI(api_key=key)
        model = model_override or "gpt-4o-mini"
    else:
        raise SystemExit(f"unknown backend {backend}")
    return client, model


def call_llm(client, model, prompt: str) -> str:
    try:
        comp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return comp.choices[0].message.content or ""
    except Exception as e:
        print(f"[judge] error: {e}", file=sys.stderr)
        return ""


def score_one_file(path: Path, client, model, max_workers: int) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    case = list(data.keys())[0]
    record_list = data[case]["precision"]["record_list"]
    metrics = data[case]["precision"].get("metrics", {})
    print(f"\n[judge] {path.name}: {len(record_list)} records (IR metrics={metrics})")

    prompts = [
        build_score_prompt(r["true_answer"], r["pred_answer"], r["question"])
        for r in record_list
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        responses = list(
            tqdm(
                ex.map(lambda p: call_llm(client, model, p), prompts),
                total=len(prompts),
                desc=path.stem,
            )
        )

    scores = [parse_score(r) for r in responses]
    avg = float(np.mean(scores)) if scores else 0.0
    out = {
        "bench": path.stem,
        "n_scored": len(scores),
        "avg_llm_score_0_5": round(avg, 4),
        "ir_metrics": metrics,
        "score_distribution": {
            str(s): scores.count(s) for s in range(6)
        },
        "records": [
            {
                "question": r["question"],
                "true_answer": r["true_answer"],
                "pred_answer": r["pred_answer"],
                "score": s,
            }
            for r, s in zip(record_list, scores)
        ],
    }
    out_path = path.with_name(f"{path.stem}_llmscore.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[judge] -> {out_path}  avg={avg:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        default="/workspace/eval_evermind_aligned",
        help="dir containing per-bench {bench}.json files",
    )
    ap.add_argument(
        "--backend",
        default="openrouter",
        choices=["openrouter", "openai"],
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--benches", default="", help="comma list; empty = all")
    args = ap.parse_args()

    client, model = make_client_and_model(args.backend, args.model)
    print(f"[judge] backend={args.backend}  model={model}")

    in_dir = Path(args.in_dir)
    if args.benches:
        files = [in_dir / f"{b.strip()}.json" for b in args.benches.split(",") if b.strip()]
    else:
        files = [
            Path(p)
            for p in glob(str(in_dir / "*.json"))
            if "summary" not in Path(p).stem and "llmscore" not in Path(p).stem
        ]
    print(f"[judge] files={[f.name for f in files]}")

    summary = {}
    for f in files:
        if not f.exists():
            print(f"[judge] skip missing {f}")
            continue
        result = score_one_file(f, client, model, args.max_workers)
        summary[result["bench"]] = result["avg_llm_score_0_5"]

    avg_all = float(np.mean(list(summary.values()))) if summary else 0.0
    summary["AVERAGE"] = round(avg_all, 4)
    out_summary = in_dir / "llmscore_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "model": model, "scores": summary}, f, indent=2)
    print(f"\n========== LLM-judge summary ==========")
    for k, v in summary.items():
        print(f"  {k:25s}  {v}")
    print(f"\n  -> {out_summary}")


if __name__ == "__main__":
    main()
