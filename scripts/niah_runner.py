"""Needle-in-a-Haystack (NIAH) evaluation for the MSA inference pipeline.

Protocol
--------
1. Load a pre-encoded haystack corpus (e.g. triviaqa_10M, ~11M tokens).
2. Encode N synthetic "needle" documents on the fly. Each needle is a short
   factoid like ``The secret access code for vault Z42 is BANANA-7423.``,
   paired with a question (``What is the secret access code for vault Z42?``)
   and a unique answer.
3. For each depth p in ``--depths``:
   a. Splice the encoded needle into each MSA layer's corpus tensor at
      doc position ``int(p * n_docs)``.
   b. Run :meth:`SparseGenerator.retrieve` -> check if the needle's
      position is in router's top-k (router accuracy).
   c. Run :meth:`SparseGenerator.generate` -> check if the needle's
      answer string appears in the LM output (LM accuracy).
4. Aggregate over the N needles and write ``out_json``.

Run:
    python scripts/niah_runner.py \
        --qwen_path /workspace/qwen35_base \
        --ckpt /workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt \
        --encoded_dir /workspace/encoded_corpora/triviaqa_10M \
        --depths 0,0.25,0.5,0.75,1.0 \
        --n_needles 6 \
        --out_json /workspace/logs/niah_10M.json
"""
import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_runner import build_model

from msa.inference.offline_encoder import EncoderConfig, _DocDataset, _collate
from msa.inference.router_engine import load_encoded_corpus, EncodedCorpus
from msa.inference.sparse_generator import SparseGenerator
from msa.inference.prompt_template import (
    build_msa_train_prompt, parse_msa_train_response, exact_match, text_f1,
)


@dataclass
class Needle:
    fact_text: str
    question: str
    answer: str


_NEEDLE_TEMPLATES = [
    ("The secret access code for vault {tag} is {ans}.",
     "What is the secret access code for vault {tag}?"),
    ("Operative {tag}'s callsign in the briefing is {ans}.",
     "What is operative {tag}'s callsign?"),
    ("The Foundation grant ID for project {tag} is {ans}.",
     "What is the Foundation grant ID for project {tag}?"),
    ("Sample {tag} was tagged with the laboratory marker {ans}.",
     "Which laboratory marker was used to tag sample {tag}?"),
    ("Container {tag} is sealed with the tamper-evident code {ans}.",
     "What is the tamper-evident code on container {tag}?"),
    ("In the museum catalog, exhibit {tag} carries the inventory number {ans}.",
     "What is the inventory number for exhibit {tag} in the museum catalog?"),
]


def make_needles(n: int, seed: int = 0) -> list[Needle]:
    rng = random.Random(seed)
    out: list[Needle] = []
    for i in range(n):
        fact_tpl, q_tpl = _NEEDLE_TEMPLATES[i % len(_NEEDLE_TEMPLATES)]
        tag = "Z" + str(40 + i)
        # Distinctive answer string the LM is unlikely to hallucinate
        ans = "{}-{:04d}".format(
            "".join(rng.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", 6)),
            rng.randint(1000, 9999),
        )
        out.append(
            Needle(
                fact_text=fact_tpl.format(tag=tag, ans=ans),
                question=q_tpl.format(tag=tag),
                answer=ans,
            )
        )
    return out


@torch.inference_mode()
def encode_one_doc(
    model, tokenizer, doc_text: str, max_doc_len: int, device: str,
    target_dtype: torch.dtype,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Encode a single doc and return ``{layer: (K_chunk, V_chunk, KR_chunk)}``
    where each tensor has shape ``(n_chunks, n_heads, head_dim)`` (no batch axis).
    """
    ds = _DocDataset([doc_text], tokenizer, max_doc_len)
    batch = _collate([ds[0]])
    doc_input_ids = batch["input_ids"].to(device).unsqueeze(0)
    doc_attention_mask = batch["attention_mask"].to(device).unsqueeze(0)
    pooled = model.model.encode_docs(
        doc_input_ids=doc_input_ids, doc_attention_mask=doc_attention_mask,
    )
    out: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for layer_idx, (K_bar, V_bar, Kr_bar) in pooled.items():
        K = K_bar[0, 0].to(target_dtype)
        V = V_bar[0, 0].to(target_dtype)
        KR = Kr_bar[0, 0].to(target_dtype)
        out[layer_idx] = (K, V, KR)
    return out


def splice_needle(
    base: dict[int, EncodedCorpus],
    needle: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    position: int,
    new_doc_id: int = -42,
) -> dict[int, EncodedCorpus]:
    out: dict[int, EncodedCorpus] = {}
    for layer_idx, ec in base.items():
        K, V, KR = needle[layer_idx]
        out[layer_idx] = ec.insert_doc(
            K, V, KR, position=position, new_doc_id=new_doc_id,
        )
    return out


def needle_in_topk(
    pred_doc_ids: list[int], needle_pos: int, base_n_docs: int,
) -> bool:
    """The needle's ROW-index in the spliced corpus is ``needle_pos``.
    But pred_doc_ids returned by SparseGenerator are global doc_ids
    (i.e. ``ec.doc_ids[row]``). We assigned the needle ``new_doc_id=-42``,
    so we just look for that sentinel.
    """
    return -42 in pred_doc_ids


def run_one_needle(
    sg: SparseGenerator,
    base_corpus: dict[int, EncodedCorpus],
    needle_pooled: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    needle: Needle,
    depths: list[float],
    k_router_topk: int,
    max_new_tokens: int,
) -> dict:
    base_n_docs = next(iter(base_corpus.values())).n_docs
    print(f"\n--- needle: {needle.question}")
    print(f"    expect: {needle.answer}")
    rows = []
    for depth in depths:
        pos = max(0, min(base_n_docs, int(round(depth * base_n_docs))))
        spliced = splice_needle(base_corpus, needle_pooled, position=pos, new_doc_id=-42)
        sg_d = SparseGenerator(
            model=sg.model, tokenizer=sg.tokenizer,
            encoded_corpus=spliced, device=sg.device,
        )

        torch.cuda.synchronize()
        t0 = time.time()
        r = sg_d.retrieve(query_text=needle.question, k_docs=k_router_topk, agg="mean")
        torch.cuda.synchronize()
        retr_t = time.time() - t0

        retrieved = needle_in_topk(r.pred_doc_ids, pos, base_n_docs)

        torch.cuda.synchronize()
        t0 = time.time()
        prompt = build_msa_train_prompt(needle.question)
        g = sg_d.generate(
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            stop_strings=["<|im_end|>"],
            temperature=0.0,
            use_kv_cache=True,
        )
        torch.cuda.synchronize()
        gen_t = time.time() - t0

        _, answer, dbg = parse_msa_train_response(g["text"])
        em = exact_match(answer, needle.answer)
        f1 = text_f1(answer, needle.answer)
        contains = needle.answer.lower() in (g["text"] or "").lower()

        rows.append({
            "depth": depth, "position": pos,
            "retrieved": bool(retrieved),
            "router_topk": r.pred_doc_ids[: min(8, k_router_topk)],
            "answer": answer[:120],
            "em": em, "text_f1": f1, "answer_contains": contains,
            "retrieve_sec": retr_t, "generate_sec": gen_t,
            "n_new_tokens": g["n_new_tokens"],
            "stopped_by": g["stopped_by"],
        })
        print(
            f"   depth={depth:.2f} pos={pos:6d}/{base_n_docs}  "
            f"retr={'Y' if retrieved else '.'} contains={'Y' if contains else '.'} "
            f"em={em:.0f} f1={f1:.2f}  pred={answer[:48]!r} "
            f"gen={gen_t:.1f}s"
        )

        del spliced, sg_d
        torch.cuda.empty_cache()
    return {
        "needle": {"question": needle.question, "answer": needle.answer,
                   "fact": needle.fact_text},
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_path", default="/workspace/qwen35_base")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoded_dir", required=True,
                    help="dir produced by encode_triviaqa_10m.py")
    ap.add_argument("--max_doc_len", type=int, default=384,
                    help="must match the encoder's max_doc_len for shape compat")
    ap.add_argument("--depths", default="0,0.25,0.5,0.75,1.0",
                    help="comma list of needle positions in [0,1]")
    ap.add_argument("--n_needles", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_router_topk", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    depths = [float(x) for x in args.depths.split(",")]
    print(f"[niah] depths={depths}")
    print(f"[niah] loading model qwen={args.qwen_path}, ckpt={args.ckpt}")
    t0 = time.time()
    model, tokenizer = build_model(args.qwen_path, args.ckpt, device="cuda:0")
    print(f"[niah] model loaded in {time.time()-t0:.1f}s")

    print(f"[niah] loading encoded haystack from {args.encoded_dir}")
    base = load_encoded_corpus(args.encoded_dir, device="cuda", dtype=torch.bfloat16)
    base_n_docs = next(iter(base.values())).n_docs
    base_n_chunks_per_doc = next(iter(base.values())).n_chunks_per_doc
    print(f"[niah] haystack: {base_n_docs} docs, {base_n_chunks_per_doc} chunks/doc")

    sg = SparseGenerator(
        model=model, tokenizer=tokenizer, encoded_corpus=base, device="cuda",
    )

    needles = make_needles(args.n_needles, seed=args.seed)
    print(f"[niah] generated {len(needles)} needles")

    all_rows = []
    summaries = []
    for ni, needle in enumerate(needles):
        print(f"\n========== needle {ni+1}/{len(needles)} ==========")
        n_pooled = encode_one_doc(
            model, tokenizer, needle.fact_text,
            max_doc_len=args.max_doc_len, device="cuda",
            target_dtype=torch.bfloat16,
        )
        result = run_one_needle(
            sg, base, n_pooled, needle, depths,
            k_router_topk=args.k_router_topk,
            max_new_tokens=args.max_new_tokens,
        )
        summaries.append(result)
        all_rows.extend(
            {**row, "needle_idx": ni} for row in result["rows"]
        )

    by_depth: dict[float, list[dict]] = {}
    for row in all_rows:
        by_depth.setdefault(row["depth"], []).append(row)

    print("\n========== summary ==========")
    print(f"{'depth':>6} {'retr':>6} {'contains':>9} {'em':>6} {'text_f1':>8}")
    summary_table = {}
    for d in depths:
        rows = by_depth.get(d, [])
        if not rows:
            continue
        n = len(rows)
        retr = sum(1 for r in rows if r["retrieved"]) / n
        cont = sum(1 for r in rows if r["answer_contains"]) / n
        em = sum(r["em"] for r in rows) / n
        f1 = sum(r["text_f1"] for r in rows) / n
        print(f"{d:6.2f} {retr*100:5.1f}% {cont*100:8.1f}% {em*100:5.1f}% {f1*100:7.1f}%")
        summary_table[str(d)] = {
            "n": n, "router_recall": retr, "answer_contains": cont,
            "em": em, "text_f1": f1,
        }

    payload = {
        "haystack_dir": args.encoded_dir,
        "haystack_n_docs": base_n_docs,
        "haystack_n_chunks_per_doc": base_n_chunks_per_doc,
        "depths": depths,
        "n_needles": args.n_needles,
        "k_router_topk": args.k_router_topk,
        "max_new_tokens": args.max_new_tokens,
        "summary_by_depth": summary_table,
        "details": summaries,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[niah] wrote {args.out_json}")


if __name__ == "__main__":
    main()
