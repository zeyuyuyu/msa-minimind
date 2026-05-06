"""RULER-style Needle-in-a-Haystack runner for the MSA inference pipeline.

Implements RULER (NVIDIA, 2024) 8 standard NIAH variants:
  niah_s1   single needle, magic-number value          (RULER niah_single_1)
  niah_s2   single needle, type-marker value           (RULER niah_single_2)
  niah_s3   single needle, UUID-style value            (RULER niah_single_3)
  niah_mk1  4 distractor needles + 1 target            (RULER niah_multikey_1)
  niah_mk2  8 distractor needles + 1 target            (RULER niah_multikey_2)
  niah_mk3  16 distractor needles + 1 target           (RULER niah_multikey_3)
  niah_mv   1 key with 4 values across 4 docs          (RULER niah_multivalue)
  niah_mq   4 needles, query each independently        (RULER niah_multiquery)

For each task we report:
  router_recall : fraction of trials where target needle's chunk made router top-k
  answer_em     : exact match between LM output and gold answer
  answer_f1     : token F1
  answer_contains : gold answer substring present in LM output (loose metric)
  multivalue_recall (mv only): fraction of values returned per query
  multiquery_recall (mq only): fraction of distinct needles correctly answered

Run:
  python scripts/niah_ruler_runner.py \
    --qwen_path /workspace/qwen35_base \
    --ckpt     /workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt \
    --encoded_dir /workspace/encoded_corpora/ms_100M_full \
    --tasks niah_s1,niah_s2,niah_s3,niah_mk1,niah_mk2,niah_mk3,niah_mv,niah_mq \
    --depths 0.0,0.25,0.5,0.75,1.0 \
    --n_trials 20 \
    --out_json /workspace/logs/ruler_100M.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_runner import build_model

from msa.inference.offline_encoder import _DocDataset, _collate
from msa.inference.router_engine import EncodedCorpus, load_encoded_corpus
from msa.inference.sparse_generator import SparseGenerator
from msa.inference.prompt_template import (
    build_msa_train_prompt, parse_msa_train_response, exact_match, text_f1,
)


# ---------------------------------------------------------------- needles ----

@dataclass
class Needle:
    fact_text: str
    question: str
    answer: str
    sentinel: int = -42


# Common natural-language anchors (RULER-style "essay" framing). The router
# operates on doc-level latent KV so we keep needles ~1-3 sentences each.
_S1_TEMPLATES = [
    ("One of the special magic numbers for {tag} is: {ans}.",
     "What is the special magic number for {tag} mentioned in the provided text?"),
    ("The magic number associated with {tag} is: {ans}.",
     "Recall the magic number associated with {tag}."),
    ("Note: the secret magic number bound to {tag} is: {ans}.",
     "What magic number is bound to {tag}?"),
    ("Remember, the magic identifier for {tag} is {ans}.",
     "What magic identifier was assigned to {tag}?"),
    ("For reference, the special magic value for {tag} is {ans}.",
     "Find the special magic value for {tag}."),
    ("In the registry, {tag} is recorded with magic number {ans}.",
     "What magic number is registered for {tag}?"),
]


def _rand_alpha(rng: random.Random, n: int) -> str:
    return "".join(rng.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", n))


_RULER_CITIES = [
    "Singapore", "Reykjavik", "Helsinki", "Vancouver", "Wellington",
    "Casablanca", "Kyoto", "Lisbon", "Tallinn", "Oslo", "Marrakech",
    "Bangkok", "Vienna", "Edinburgh", "Auckland", "Stockholm",
]


def _rand_value(rng: random.Random, variant: str) -> str:
    """Mint paper-aligned RULER magic-number needles (digit-only -> LM friendly).

    s1 -> single 7-digit magic number          (e.g. "8345783")          paper niah_single_1
    s2 -> city + 7-digit magic number          (e.g. "Singapore: 4567891") paper niah_single_2
    s3 -> 3-element magic number chain         (e.g. "9123456-7891234-5671234") paper niah_single_3
    """
    if variant == "s1":
        return f"{rng.randint(1_000_000, 9_999_999)}"
    if variant == "s2":
        city = rng.choice(_RULER_CITIES)
        return f"{city}: {rng.randint(1_000_000, 9_999_999)}"
    if variant == "s3":
        a = rng.randint(1_000_000, 9_999_999)
        b = rng.randint(1_000_000, 9_999_999)
        c = rng.randint(1_000_000, 9_999_999)
        return f"{a}-{b}-{c}"
    raise ValueError(variant)


def make_needle(idx: int, rng: random.Random, variant: str = "s1") -> Needle:
    fact_tpl, q_tpl = _S1_TEMPLATES[idx % len(_S1_TEMPLATES)]
    tag = "Z" + str(40 + idx)
    ans = _rand_value(rng, variant)
    return Needle(
        fact_text=fact_tpl.format(tag=tag, ans=ans),
        question=q_tpl.format(tag=tag),
        answer=ans,
        sentinel=-(1000 + idx),  # unique sentinel per needle
    )


def make_multivalue_needles(n_values: int, rng: random.Random) -> tuple[list[Needle], str, list[str]]:
    """1 key shared across N docs, each holding a different magic number.

    Paper-aligned RULER niah_multivalue: one tag, multiple digit-only values
    scattered across N docs. Query asks for ALL values.

    Returns (needle_docs, query, gold_values).
    """
    fact_tpl = "One of the special magic numbers for {tag} is: {ans}."
    q_tpl = "List all special magic numbers for {tag} mentioned in the text."
    tag = "Z" + str(rng.randint(40, 99))
    needles = []
    values = []
    for i in range(n_values):
        ans = f"{rng.randint(1_000_000, 9_999_999)}"
        values.append(ans)
        needles.append(Needle(
            fact_text=fact_tpl.format(tag=tag, ans=ans),
            question="",
            answer=ans,
            sentinel=-(2000 + i),
        ))
    query = q_tpl.format(tag=tag)
    return needles, query, values


# ---------------------------------------------------------------- splice ----

@torch.inference_mode()
def encode_needle(
    model, tokenizer, doc_text: str, max_doc_len: int, device: str,
    dtype: torch.dtype,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ds = _DocDataset([doc_text], tokenizer, max_doc_len)
    batch = _collate([ds[0]])
    doc_input_ids = batch["input_ids"].to(device).unsqueeze(0)
    doc_attention_mask = batch["attention_mask"].to(device).unsqueeze(0)
    pooled = model.model.encode_docs(
        doc_input_ids=doc_input_ids, doc_attention_mask=doc_attention_mask,
    )
    out = {}
    for layer_idx, (K_bar, V_bar, Kr_bar) in pooled.items():
        out[layer_idx] = (
            K_bar[0, 0].to(dtype),
            V_bar[0, 0].to(dtype),
            Kr_bar[0, 0].to(dtype),
        )
    return out


def splice_many(
    base: dict[int, EncodedCorpus],
    needle_pooleds: list[dict[int, tuple]],
    positions: list[int],
    sentinels: list[int],
) -> dict[int, EncodedCorpus]:
    """Splice multiple needles. Positions are RELATIVE TO ORIGINAL corpus -
    we sort and apply right-to-left so earlier indices stay valid."""
    assert len(needle_pooleds) == len(positions) == len(sentinels)
    order = sorted(range(len(positions)), key=lambda i: positions[i], reverse=True)
    out = base
    for i in order:
        nd = needle_pooleds[i]
        new = {}
        for layer_idx, ec in out.items():
            K, V, KR = nd[layer_idx]
            new[layer_idx] = ec.insert_doc(
                K, V, KR, position=positions[i], new_doc_id=sentinels[i],
            )
        out = new
    return out


# ---------------------------------------------------------------- tasks ----

def _gen_answer(sg: SparseGenerator, question: str, max_new_tokens: int) -> tuple[str, str]:
    prompt = build_msa_train_prompt(question)
    g = sg.generate(
        prompt_text=prompt, max_new_tokens=max_new_tokens,
        stop_strings=["<|im_end|>"], temperature=0.0, use_kv_cache=True,
    )
    _, answer, _ = parse_msa_train_response(g["text"])
    return answer or "", g["text"] or ""


def run_single_task(
    sg: SparseGenerator, base: dict[int, EncodedCorpus],
    model, tokenizer, *, variant: str, n_trials: int, depths: list[float],
    k_router_topk: int, max_doc_len: int, max_new_tokens: int, seed: int,
    device: str, dtype: torch.dtype,
) -> dict:
    rng = random.Random(seed)
    n_docs = next(iter(base.values())).n_docs
    rows = []
    for trial in range(n_trials):
        needle = make_needle(trial, rng, variant=variant)
        nd = encode_needle(model, tokenizer, needle.fact_text, max_doc_len, device, dtype)
        depth = depths[trial % len(depths)]
        pos = max(0, min(n_docs, int(round(depth * n_docs))))
        spliced = splice_many(base, [nd], [pos], [needle.sentinel])
        sg_d = SparseGenerator(model=sg.model, tokenizer=sg.tokenizer,
                               encoded_corpus=spliced, device=sg.device)
        r = sg_d.retrieve(query_text=needle.question, k_docs=k_router_topk, agg="mean")
        retrieved = needle.sentinel in r.pred_doc_ids
        ans, raw = _gen_answer(sg_d, needle.question, max_new_tokens)
        em = exact_match(ans, needle.answer)
        f1 = text_f1(ans, needle.answer)
        contains = needle.answer.lower() in raw.lower()
        rows.append({
            "trial": trial, "depth": depth, "pos": pos,
            "retrieved": bool(retrieved), "em": em, "f1": f1,
            "contains": bool(contains), "answer": ans[:80],
            "expect": needle.answer,
        })
        del spliced, sg_d
        torch.cuda.empty_cache()
        print(f"  [{variant} {trial+1}/{n_trials}] d={depth:.2f} retr={'Y' if retrieved else '.'} "
              f"em={em:.0f} f1={f1:.2f} contains={'Y' if contains else '.'} pred={ans[:32]!r}")
    return _summarize_rows(rows, depths)


def run_multikey_task(
    sg: SparseGenerator, base: dict[int, EncodedCorpus],
    model, tokenizer, *, n_distractors: int, n_trials: int, depths: list[float],
    k_router_topk: int, max_doc_len: int, max_new_tokens: int, seed: int,
    device: str, dtype: torch.dtype,
) -> dict:
    """Multi-key NIAH: insert 1 target needle + N distractor needles into haystack.
    Query asks about target's tag specifically, distractors have other tags."""
    rng = random.Random(seed + 1000)
    n_docs = next(iter(base.values())).n_docs
    rows = []
    for trial in range(n_trials):
        target = make_needle(trial, rng, variant="s1")
        distractors = [make_needle(100 + trial * 100 + j, rng, variant="s1")
                       for j in range(n_distractors)]
        all_needles = [target] + distractors
        depth = depths[trial % len(depths)]
        target_pos = max(0, min(n_docs, int(round(depth * n_docs))))
        positions = [target_pos]
        rng2 = random.Random(seed + trial * 7919)
        for _ in distractors:
            positions.append(rng2.randint(0, n_docs))
        sentinels = [n.sentinel for n in all_needles]
        nds = [encode_needle(model, tokenizer, n.fact_text, max_doc_len, device, dtype)
               for n in all_needles]
        spliced = splice_many(base, nds, positions, sentinels)
        sg_d = SparseGenerator(model=sg.model, tokenizer=sg.tokenizer,
                               encoded_corpus=spliced, device=sg.device)
        r = sg_d.retrieve(query_text=target.question, k_docs=k_router_topk, agg="mean")
        retrieved = target.sentinel in r.pred_doc_ids
        ans, raw = _gen_answer(sg_d, target.question, max_new_tokens)
        em = exact_match(ans, target.answer)
        f1 = text_f1(ans, target.answer)
        contains = target.answer.lower() in raw.lower()
        rows.append({
            "trial": trial, "depth": depth, "pos": target_pos,
            "n_distractors": n_distractors,
            "retrieved": bool(retrieved), "em": em, "f1": f1,
            "contains": bool(contains), "answer": ans[:80],
            "expect": target.answer,
        })
        del spliced, sg_d
        torch.cuda.empty_cache()
        print(f"  [mk{n_distractors} {trial+1}/{n_trials}] d={depth:.2f} retr={'Y' if retrieved else '.'} "
              f"em={em:.0f} f1={f1:.2f} contains={'Y' if contains else '.'} pred={ans[:32]!r}")
    return _summarize_rows(rows, depths, extra_keys=["n_distractors"])


def run_multivalue_task(
    sg: SparseGenerator, base: dict[int, EncodedCorpus],
    model, tokenizer, *, n_values: int, n_trials: int, depths: list[float],
    k_router_topk: int, max_doc_len: int, max_new_tokens: int, seed: int,
    device: str, dtype: torch.dtype,
) -> dict:
    """1 key with multiple values, each in a separate doc. Query asks for all."""
    rng = random.Random(seed + 2000)
    n_docs = next(iter(base.values())).n_docs
    rows = []
    for trial in range(n_trials):
        needles, query, gold_values = make_multivalue_needles(n_values, rng)
        rng2 = random.Random(seed + trial * 9931)
        positions = []
        for i in range(n_values):
            base_d = depths[i % len(depths)]
            jitter = (i - n_values / 2) * 0.05
            d = max(0.0, min(1.0, base_d + jitter))
            positions.append(max(0, min(n_docs, int(round(d * n_docs)))))
        sentinels = [n.sentinel for n in needles]
        nds = [encode_needle(model, tokenizer, n.fact_text, max_doc_len, device, dtype)
               for n in needles]
        spliced = splice_many(base, nds, positions, sentinels)
        sg_d = SparseGenerator(model=sg.model, tokenizer=sg.tokenizer,
                               encoded_corpus=spliced, device=sg.device)
        r = sg_d.retrieve(query_text=query, k_docs=k_router_topk, agg="mean")
        retrieved_set = set(r.pred_doc_ids.tolist() if hasattr(r.pred_doc_ids, "tolist")
                            else list(r.pred_doc_ids))
        retr_n = sum(1 for s in sentinels if s in retrieved_set)
        retr_recall = retr_n / n_values
        ans, raw = _gen_answer(sg_d, query, max_new_tokens)
        ans_lower = (raw or "").lower()
        gold_hit = sum(1 for v in gold_values if v.lower() in ans_lower)
        gold_recall = gold_hit / n_values
        rows.append({
            "trial": trial, "n_values": n_values,
            "retrieved_n": retr_n, "retrieved_recall": retr_recall,
            "answer_recall": gold_recall, "answer_hit": gold_hit,
            "answer": ans[:120], "expect": gold_values,
        })
        del spliced, sg_d
        torch.cuda.empty_cache()
        print(f"  [mv {trial+1}/{n_trials}] retr={retr_n}/{n_values} ans_hit={gold_hit}/{n_values} pred={ans[:48]!r}")
    n = len(rows)
    by_overall = {
        "n_trials": n,
        "router_recall_mean": sum(r["retrieved_recall"] for r in rows) / max(1, n),
        "answer_recall_mean": sum(r["answer_recall"] for r in rows) / max(1, n),
        "answer_full_hit_rate": sum(1 for r in rows if r["answer_recall"] == 1.0) / max(1, n),
    }
    return {"by_depth": {}, "overall": by_overall, "rows": rows}


def run_multiquery_task(
    sg: SparseGenerator, base: dict[int, EncodedCorpus],
    model, tokenizer, *, n_queries: int, n_trials: int, depths: list[float],
    k_router_topk: int, max_doc_len: int, max_new_tokens: int, seed: int,
    device: str, dtype: torch.dtype,
) -> dict:
    """Insert N needles into haystack, then query each separately."""
    rng = random.Random(seed + 3000)
    n_docs = next(iter(base.values())).n_docs
    rows = []
    for trial in range(n_trials):
        needles = [make_needle(200 + trial * 100 + j, rng, variant="s1")
                   for j in range(n_queries)]
        positions = []
        for i in range(n_queries):
            base_d = depths[i % len(depths)]
            positions.append(max(0, min(n_docs, int(round(base_d * n_docs)))))
        sentinels = [n.sentinel for n in needles]
        nds = [encode_needle(model, tokenizer, n.fact_text, max_doc_len, device, dtype)
               for n in needles]
        spliced = splice_many(base, nds, positions, sentinels)
        sg_d = SparseGenerator(model=sg.model, tokenizer=sg.tokenizer,
                               encoded_corpus=spliced, device=sg.device)
        sub_results = []
        for q_idx, n_target in enumerate(needles):
            r = sg_d.retrieve(query_text=n_target.question, k_docs=k_router_topk, agg="mean")
            retrieved = n_target.sentinel in r.pred_doc_ids
            ans, raw = _gen_answer(sg_d, n_target.question, max_new_tokens)
            em = exact_match(ans, n_target.answer)
            sub_results.append({
                "q_idx": q_idx, "depth": depths[q_idx % len(depths)],
                "retrieved": bool(retrieved), "em": em,
                "contains": n_target.answer.lower() in (raw or "").lower(),
            })
        del spliced, sg_d
        torch.cuda.empty_cache()
        retr_recall = sum(1 for s in sub_results if s["retrieved"]) / n_queries
        em_rate = sum(s["em"] for s in sub_results) / n_queries
        rows.append({"trial": trial, "n_queries": n_queries,
                     "retrieved_recall": retr_recall, "em_rate": em_rate,
                     "subs": sub_results})
        print(f"  [mq {trial+1}/{n_trials}] retr_recall={retr_recall:.2f} em_rate={em_rate:.2f}")
    n = len(rows)
    by_overall = {
        "n_trials": n,
        "router_recall_mean": sum(r["retrieved_recall"] for r in rows) / max(1, n),
        "em_rate_mean": sum(r["em_rate"] for r in rows) / max(1, n),
    }
    return {"by_depth": {}, "overall": by_overall, "rows": rows}


# -------------------------------------------------------------- aggregate ---

def _summarize_rows(rows: list[dict], depths: list[float],
                    extra_keys: list[str] | None = None) -> dict:
    by_depth = {}
    for d in depths:
        rs = [r for r in rows if r["depth"] == d]
        if not rs:
            continue
        n = len(rs)
        by_depth[str(d)] = {
            "n": n,
            "router_recall": sum(1 for r in rs if r["retrieved"]) / n,
            "em": sum(r["em"] for r in rs) / n,
            "f1": sum(r["f1"] for r in rs) / n,
            "contains": sum(1 for r in rs if r["contains"]) / n,
        }
    n = len(rows)
    overall = {
        "n_trials": n,
        "router_recall": sum(1 for r in rows if r["retrieved"]) / max(1, n),
        "em": sum(r["em"] for r in rows) / max(1, n),
        "f1": sum(r["f1"] for r in rows) / max(1, n),
        "contains": sum(1 for r in rows if r["contains"]) / max(1, n),
    }
    return {"by_depth": by_depth, "overall": overall, "rows": rows}


# ----------------------------------------------------------------- main ----

TASKS = {
    "niah_s1":  ("single",     {"variant": "s1"}),
    "niah_s2":  ("single",     {"variant": "s2"}),
    "niah_s3":  ("single",     {"variant": "s3"}),
    "niah_mk1": ("multikey",   {"n_distractors": 4}),
    "niah_mk2": ("multikey",   {"n_distractors": 8}),
    "niah_mk3": ("multikey",   {"n_distractors": 16}),
    "niah_mv":  ("multivalue", {"n_values": 4}),
    "niah_mq":  ("multiquery", {"n_queries": 4}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_path", default="/workspace/qwen35_base")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoded_dir", required=True)
    ap.add_argument("--max_doc_len", type=int, default=384)
    ap.add_argument("--depths", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--n_trials", type=int, default=10,
                    help="trials per task (single/multikey: depths cycled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_router_topk", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--tasks", default=",".join(TASKS.keys()))
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    depths = [float(x) for x in args.depths.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in TASKS:
            raise ValueError(f"unknown task {t!r}; valid: {list(TASKS)}")

    print(f"[ruler] depths={depths}  n_trials={args.n_trials}  tasks={tasks}")
    print(f"[ruler] loading model qwen={args.qwen_path}, ckpt={args.ckpt}")
    t0 = time.time()
    model, tokenizer = build_model(args.qwen_path, args.ckpt, device="cuda:0")
    print(f"[ruler] model loaded in {time.time()-t0:.1f}s")

    print(f"[ruler] loading encoded haystack from {args.encoded_dir}")
    base = load_encoded_corpus(args.encoded_dir, device="cuda", dtype=torch.bfloat16)
    n_docs = next(iter(base.values())).n_docs
    n_chunks_per_doc = next(iter(base.values())).n_chunks_per_doc
    # Encoder runs with msa_chunk_size=32 in our SFT-S2 config, so each chunk
    # is 32 tokens. The needle doc must produce *exactly* n_chunks_per_doc
    # chunks to match the haystack tensor layout, otherwise insert_doc raises.
    chunk_size = 32
    inferred_max_doc_len = n_chunks_per_doc * chunk_size
    if args.max_doc_len != inferred_max_doc_len:
        print(f"[ruler] override --max_doc_len {args.max_doc_len} -> {inferred_max_doc_len} "
              f"(haystack n_chunks_per_doc={n_chunks_per_doc} * chunk_size={chunk_size})")
        args.max_doc_len = inferred_max_doc_len
    print(f"[ruler] haystack: {n_docs} docs, {n_chunks_per_doc} chunks/doc, "
          f"max_doc_len={args.max_doc_len}, "
          f"~{n_docs * args.max_doc_len / 1e6:.1f}M tokens")

    sg = SparseGenerator(model=model, tokenizer=tokenizer,
                         encoded_corpus=base, device="cuda")

    common = dict(
        n_trials=args.n_trials, depths=depths,
        k_router_topk=args.k_router_topk, max_doc_len=args.max_doc_len,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
        device="cuda", dtype=torch.bfloat16,
    )

    out_per_task = {}
    for task in tasks:
        ttype, params = TASKS[task]
        print(f"\n========== {task}  ({ttype}, {params}) ==========")
        ts = time.time()
        if ttype == "single":
            res = run_single_task(sg, base, model, tokenizer, **params, **common)
        elif ttype == "multikey":
            res = run_multikey_task(sg, base, model, tokenizer, **params, **common)
        elif ttype == "multivalue":
            res = run_multivalue_task(sg, base, model, tokenizer, **params, **common)
        elif ttype == "multiquery":
            res = run_multiquery_task(sg, base, model, tokenizer, **params, **common)
        else:
            raise RuntimeError(ttype)
        res["task"] = task
        res["task_type"] = ttype
        res["task_params"] = params
        res["wall_seconds"] = time.time() - ts
        out_per_task[task] = res
        print(f"-- {task} overall: {res['overall']}  ({res['wall_seconds']:.1f}s)")

    print("\n========== RULER summary ==========")
    print(f"{'task':<10} {'router':>8} {'em':>6} {'f1':>6} {'contains':>10}")
    for task, res in out_per_task.items():
        ov = res["overall"]
        rr = ov.get("router_recall") or ov.get("router_recall_mean") or 0.0
        em = ov.get("em") or ov.get("em_rate_mean") or ov.get("answer_recall_mean") or 0.0
        f1 = ov.get("f1") or 0.0
        co = ov.get("contains") or ov.get("answer_full_hit_rate") or 0.0
        print(f"{task:<10} {rr*100:7.1f}% {em*100:5.1f}% {f1*100:5.1f}% {co*100:9.1f}%")

    payload = {
        "encoded_dir": args.encoded_dir,
        "n_docs": n_docs,
        "n_chunks_per_doc": n_chunks_per_doc,
        "approx_tokens": n_docs * n_chunks_per_doc * 64,
        "depths": depths,
        "n_trials": args.n_trials,
        "k_router_topk": args.k_router_topk,
        "max_new_tokens": args.max_new_tokens,
        "tasks": out_per_task,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[ruler] wrote {args.out_json}")


if __name__ == "__main__":
    main()
