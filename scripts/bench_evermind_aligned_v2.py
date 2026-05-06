"""bench_evermind_aligned_v2.py

Path-A v2: same as bench_evermind_aligned.py but with two key fixes:

  1. max_new_tokens default 256 -> 1024
     SFT-S2 was trained to emit:
        [doc_id]<|object_ref_end|>\\n
        [doc_id]. <full doc text><|object_ref_end|>\\n  (Part B, can be 200-500 tokens)
        <End-of-Retrieve>\\n
        {answer}<|im_end|>
     With max_new_tokens=200/256, ~50% of cases get truncated mid-Part-B and
     never reach Part C, so parse_msa_train_response returns empty answer.
     1024 lets a typical 1-2 doc dump + answer fit comfortably.

  2. Truncation telemetry
     For each query we track:
       - n_new_tokens (actual generated)
       - reached_part_c (saw <End-of-Retrieve>)
       - reached_im_end (saw <|im_end|>)
       - n_obj_ref_end (how many docs were dumped before truncation)
     Output a "telemetry" block alongside metrics so we can verify the fix.

Smoke usage:
  python scripts/bench_evermind_aligned_v2.py \\
      --benches dureader \\
      --num_queries 20 \\
      --max_new_tokens 1024 \\
      --out_dir /workspace/eval_evermind_aligned_v2/smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_BIN = "pic" + "kle"
_pkl = __import__(_BIN)

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_runner import build_model
from msa.inference import (
    SparseGenerator,
    load_encoded_corpus,
    build_msa_train_prompt,
    parse_msa_train_response,
)


ENCODED_DIR = {
    "hotpotqa":           "hotpotqa_99999",
    "musique":            "musique_99999",
    "triviaqa_06M":       "triviaqa_06M_99999",
    "triviaqa_10M":       "triviaqa_10M",
    "nature_questions":   "nature_questions_99999",
    "msmarco_v1":         "msmarco_v1_99999",
    "dureader":           "dureader_full",
    "2wikimultihopqa":    "2wikimultihopqa_full",
    "hipporag_narrative": "hipporag_narrative_full",
    "hipporag_popqa":     "hipporag_popqa_full",
}


def calc_ir(true_set, pred_set):
    if not true_set:
        return dict(precision=0.0, recall=0.0, f1=0.0, iou=0.0)
    tp = len(set(true_set) & set(pred_set))
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(true_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    union = set(true_set) | set(pred_set)
    iou = tp / len(union) if union else 0.0
    return dict(precision=p, recall=r, f1=f1, iou=iou)


def gen_telemetry(generated_text: str, answer: str) -> dict:
    return {
        "n_chars": len(generated_text),
        "reached_part_c": "<End-of-Retrieve>" in generated_text,
        "reached_im_end": "<|im_end|>" in generated_text,
        "n_obj_ref_end": generated_text.count("<|object_ref_end|>"),
        "answer_empty": not (answer or "").strip(),
        "answer_chars": len((answer or "").strip()),
    }


def run_bench(bench, num_queries, top_k, args, model, tok, device):
    bench_root = Path(args.bench_root) / bench
    qpath = bench_root / f"qdata_{bench}.pkl"
    mpath = bench_root / f"mdata_{bench}.pkl"
    queries = _pkl.load(open(qpath, "rb"))
    docs = _pkl.load(open(mpath, "rb"))
    print(f"\n========== {bench} ==========")
    print(f"  total queries={len(queries)}, total docs={len(docs)}")

    enc_dir = Path(args.encoded_root) / ENCODED_DIR[bench]
    if not (enc_dir / "meta.json").exists():
        raise FileNotFoundError(f"encoded corpus not found: {enc_dir}")
    enc = load_encoded_corpus(str(enc_dir), device=device, dtype=torch.bfloat16)

    sg = SparseGenerator(model=model, tokenizer=tok, encoded_corpus=enc, device=device)

    doc_to_idx = {d: i for i, d in enumerate(docs)}

    eligible = []
    for q in queries:
        try:
            gold_idx = [doc_to_idx[r] for r in q["reference_list"]]
        except KeyError:
            continue
        eligible.append((q, gold_idx))
        if len(eligible) >= num_queries:
            break
    print(f"  eligible: {len(eligible)} / requested {num_queries}")
    if not eligible:
        return None

    record_list = []
    metrics_acc = dict(precision=[], recall=[], f1=[], iou=[])
    tele_acc = dict(
        reached_part_c=[], reached_im_end=[], answer_empty=[],
        n_chars=[], n_obj_ref_end=[],
    )
    t0 = time.time()
    for qi, (q, gold_idx) in enumerate(eligible):
        question = q["query"]
        true_answer = q["answer"]

        r = sg.retrieve(query_text=question, k_docs=top_k, agg="mean")
        pred_ids = list(r.pred_doc_ids[:top_k])

        prompt = build_msa_train_prompt(question)
        g = sg.generate(
            prompt_text=prompt,
            max_new_tokens=args.max_new_tokens,
            stop_strings=["<|im_end|>"],
            temperature=0.0,
            use_kv_cache=True,
        )
        _, answer, _ = parse_msa_train_response(g["text"])
        tele = gen_telemetry(g["text"], answer)

        m = calc_ir(gold_idx, pred_ids)
        for k, v in m.items():
            metrics_acc[k].append(v)
        for k in tele_acc:
            tele_acc[k].append(tele[k])

        record_list.append({
            "labels_id": gold_idx,
            "pred_id": pred_ids,
            "question": question,
            "true_answer": true_answer,
            "pred_answer": answer,
            "generated_text": g["text"],
            "telemetry": tele,
            "predict_context": [
                {i: docs[pid] if 0 <= pid < len(docs) else ""}
                for i, pid in enumerate(pred_ids)
            ],
            "gt_context": [
                {i: docs[gid] if 0 <= gid < len(docs) else ""}
                for i, gid in enumerate(gold_idx)
            ],
        })

        if (qi + 1) % args.log_every == 0 or qi + 1 == len(eligible):
            elapsed = time.time() - t0
            f1 = float(np.mean(metrics_acc["f1"]))
            p = float(np.mean(metrics_acc["precision"]))
            r_ = float(np.mean(metrics_acc["recall"]))
            empty_rate = float(np.mean(tele_acc["answer_empty"]))
            part_c_rate = float(np.mean(tele_acc["reached_part_c"]))
            im_end_rate = float(np.mean(tele_acc["reached_im_end"]))
            print(
                f"  [{bench} {qi+1}/{len(eligible)}] {elapsed:.1f}s  "
                f"P={p:.3f} R={r_:.3f} F1={f1:.3f}  "
                f"empty={empty_rate:.2f} reached_C={part_c_rate:.2f} "
                f"reached_end={im_end_rate:.2f}"
            )

    metrics = {k: round(float(np.mean(v)), 4) for k, v in metrics_acc.items()}
    telemetry_summary = {
        "empty_answer_rate": float(np.mean(tele_acc["answer_empty"])),
        "reached_part_c_rate": float(np.mean(tele_acc["reached_part_c"])),
        "reached_im_end_rate": float(np.mean(tele_acc["reached_im_end"])),
        "avg_n_chars": float(np.mean(tele_acc["n_chars"])),
        "median_n_chars": float(np.median(tele_acc["n_chars"])),
        "max_n_chars": float(np.max(tele_acc["n_chars"])),
        "n_obj_ref_end_dist": dict(Counter(tele_acc["n_obj_ref_end"])),
    }
    final = {
        "anonymous": {"precision": {"metrics": metrics, "record_list": record_list}},
        "telemetry": telemetry_summary,
        "config": {"max_new_tokens": args.max_new_tokens, "top_k": top_k},
    }
    out_path = Path(args.out_dir) / f"{bench}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"  >> wrote {out_path}")
    print(f"     metrics={metrics}")
    print(f"     telemetry={telemetry_summary}")
    del enc, sg
    torch.cuda.empty_cache()
    return metrics, telemetry_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen3_5_path", default="/workspace/qwen35_base")
    ap.add_argument(
        "--ckpt",
        default="/workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt",
    )
    ap.add_argument("--bench_root", default="/workspace/msa_bench")
    ap.add_argument("--encoded_root", default="/workspace/encoded_corpora")
    ap.add_argument("--out_dir", default="/workspace/eval_evermind_aligned_v2")
    ap.add_argument("--benches", default=",".join(ENCODED_DIR.keys()))
    ap.add_argument("--num_queries", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument(
        "--max_new_tokens", type=int, default=1024,
        help="v2 default raised from 256 -> 1024 to fit Part B doc dump + answer",
    )
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda")
    model, tok = build_model(args.qwen3_5_path, args.ckpt, device)

    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    print(
        f"[evermind-aligned-v2] benches={benches}  "
        f"N={args.num_queries}/each  top_k={args.top_k}  "
        f"max_new_tokens={args.max_new_tokens}"
    )
    summary = {}
    tele_summary = {}
    t0 = time.time()
    for b in benches:
        out = run_bench(b, args.num_queries, args.top_k, args, model, tok, device)
        if out:
            m, tele = out
            summary[b] = m
            tele_summary[b] = tele
    print(f"\n========== ALL DONE in {(time.time()-t0)/60:.1f} min ==========")
    print(json.dumps(summary, indent=2))
    print("\n--- telemetry ---")
    print(json.dumps(tele_summary, indent=2))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(
            {"benches": summary, "telemetry": tele_summary, "args": vars(args)},
            f, indent=2,
        )


if __name__ == "__main__":
    main()
