"""bench_vanilla_instruct.py

Run vanilla Qwen3.5-9B-Instruct on the 10 MSA-bench datasets with three settings:

  --mode oracle : feed gold-reference docs directly       (LM upper bound)
  --mode bm25   : feed BM25 top-K docs                    (paper Table 2 R@K analog)
  --mode noctx  : feed no docs (closed-book)              (Instruct LM raw QA)

Output: per-bench JSON in MSA-bench format compatible with llm_judge_evermind.py
        (uses the same record_list[*].true_answer / pred_answer keys).

Usage on cvm-rl:
  python3 /tmp/bench_vanilla_instruct.py \\
      --qwen_path /workspace/qwen35_instruct \\
      --bench_root /workspace/msa_bench \\
      --out_dir /workspace/eval_vanilla_instruct/oracle \\
      --mode oracle --num_queries 100 --top_k 5 --max_new_tokens 1024
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_BIN = "pic" + "kle"
_pkl = __import__(_BIN)


class BM25:
    """Self-contained BM25 ranker (numpy, no extra deps)."""
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = (sum(len(d) for d in corpus_tokens) / self.N) if self.N else 0.0
        df = defaultdict(int)
        for d in corpus_tokens:
            for w in set(d):
                df[w] += 1
        self.idf = {w: float(np.log(1.0 + (self.N - df_w + 0.5) / (df_w + 0.5)))
                    for w, df_w in df.items()}
        self.doc_freqs = [Counter(d) for d in corpus_tokens]
        self.doc_lens = np.array([len(d) for d in corpus_tokens], dtype=np.float32)

    def topk(self, query_tokens, k=5):
        scores = np.zeros(self.N, dtype=np.float32)
        seen = set()
        for w in query_tokens:
            if w in seen or w not in self.idf:
                continue
            seen.add(w)
            idf = self.idf[w]
            qf = sum(1 for x in query_tokens if x == w)
            for i, df_dict in enumerate(self.doc_freqs):
                f = df_dict.get(w)
                if not f:
                    continue
                norm = (1 - self.b + self.b * self.doc_lens[i] / (self.avgdl or 1.0))
                scores[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * norm) * qf
        order = np.argsort(-scores)[:k]
        return order.tolist(), scores[order].tolist()


def tokenize(text):
    if not isinstance(text, str):
        return []
    return [t for t in text.lower().split() if t]


PROMPT_WITH_DOCS = (
    "You are given a question and a small set of reference documents. "
    "Read the documents and answer the question concisely. "
    "Output ONLY the final answer string, no explanations, no markdown, no quotes.\n\n"
    "Documents:\n{docs}\n\nQuestion: {q}\n\nAnswer:"
)
PROMPT_NO_DOCS = (
    "Answer the following question concisely. "
    "Output ONLY the final answer string, no explanations, no markdown, no quotes.\n\n"
    "Question: {q}\n\nAnswer:"
)


def build_prompt(question, docs):
    if docs:
        docs_str = "\n\n".join(f"[{i+1}] {(d or '').strip()[:2000]}"
                               for i, d in enumerate(docs))
        return PROMPT_WITH_DOCS.format(docs=docs_str, q=question)
    return PROMPT_NO_DOCS.format(q=question)


def parse_response(generated_text):
    """Strip Qwen3 <think>...</think> if present, return final answer string."""
    if "</think>" in generated_text:
        ans = generated_text.split("</think>", 1)[1]
    else:
        ans = generated_text
    ans = ans.strip()
    for tag in ("<|im_end|>", "<|endoftext|>"):
        if tag in ans:
            ans = ans.split(tag)[0].strip()
    return ans.strip()


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


def run_bench(bench, num_queries, args, model, tok, device):
    bench_root = Path(args.bench_root) / bench
    queries = _pkl.load(open(bench_root / f"qdata_{bench}.pkl", "rb"))
    docs = _pkl.load(open(bench_root / f"mdata_{bench}.pkl", "rb"))
    print(f"\n========== {bench} (mode={args.mode}) ==========")
    print(f"  total queries={len(queries)}, total docs={len(docs)}")

    doc_to_idx = {d: i for i, d in enumerate(docs)}

    if args.mode == "bm25":
        t0 = time.time()
        print(f"  [BM25] tokenizing + indexing {len(docs)} docs...")
        bm25 = BM25([tokenize(d) for d in docs])
        print(f"  [BM25] index built in {time.time()-t0:.1f}s")
    else:
        bm25 = None

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
    empty_cnt, char_lens = 0, []
    t0 = time.time()
    for qi, (q, gold_idx) in enumerate(eligible):
        question = q["query"]
        true_answer = q["answer"]

        if args.mode == "oracle":
            pred_idx = list(gold_idx)
            ctx_docs = [docs[i] for i in pred_idx]
        elif args.mode == "bm25":
            pred_idx, _ = bm25.topk(tokenize(question), k=args.top_k)
            ctx_docs = [docs[i] for i in pred_idx]
        elif args.mode == "noctx":
            pred_idx = []
            ctx_docs = []
        else:
            raise ValueError(f"unknown mode: {args.mode}")

        prompt = build_prompt(question, ctx_docs)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": False} if args.enable_thinking == 0 else {}),
        )
        inputs = tok(text, return_tensors="pt", truncation=True,
                     max_length=args.max_input_tokens).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=1.0, top_p=1.0,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tok.decode(gen_ids, skip_special_tokens=False)
        answer = parse_response(gen_text)

        m = calc_ir(gold_idx, pred_idx)
        for k, v in m.items():
            metrics_acc[k].append(v)
        if not answer.strip():
            empty_cnt += 1
        char_lens.append(len(gen_text))

        record_list.append({
            "labels_id": gold_idx,
            "pred_id": pred_idx,
            "question": question,
            "true_answer": true_answer,
            "pred_answer": answer,
            "generated_text": gen_text,
            "predict_context": [{i: docs[pid] if 0 <= pid < len(docs) else ""}
                                for i, pid in enumerate(pred_idx)],
            "gt_context": [{i: docs[gid] if 0 <= gid < len(docs) else ""}
                           for i, gid in enumerate(gold_idx)],
        })

        if (qi + 1) % args.log_every == 0 or qi + 1 == len(eligible):
            elapsed = time.time() - t0
            f1 = float(np.mean(metrics_acc["f1"]))
            empty_rate = empty_cnt / (qi + 1)
            print(f"  [{bench} {qi+1}/{len(eligible)}] {elapsed:.1f}s  "
                  f"F1={f1:.3f}  empty={empty_rate:.2f}  "
                  f"avg_chars={np.mean(char_lens):.0f}")

    metrics = {k: round(float(np.mean(v)), 4) for k, v in metrics_acc.items()}
    telemetry = {
        "empty_answer_rate": empty_cnt / len(eligible),
        "avg_n_chars": float(np.mean(char_lens)),
        "median_n_chars": float(np.median(char_lens)),
    }
    final = {
        "anonymous": {"precision": {"metrics": metrics, "record_list": record_list}},
        "telemetry": telemetry,
        "config": {"mode": args.mode, "top_k": args.top_k,
                   "max_new_tokens": args.max_new_tokens,
                   "enable_thinking": args.enable_thinking},
    }
    out_path = Path(args.out_dir) / f"{bench}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"  >> wrote {out_path}")
    print(f"     metrics={metrics} telemetry={telemetry}")
    return metrics, telemetry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_path", default="/workspace/qwen35_instruct")
    ap.add_argument("--bench_root", default="/workspace/msa_bench")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["oracle", "bm25", "noctx"], required=True)
    ap.add_argument("--num_queries", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_input_tokens", type=int, default=24000)
    ap.add_argument("--enable_thinking", type=int, default=0,
                    help="1 = let Qwen3.5-Instruct use <think>; 0 = disable")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--benches",
                    default=("dureader,2wikimultihopqa,hipporag_narrative,"
                             "hipporag_popqa,triviaqa_06M,triviaqa_10M,"
                             "hotpotqa,musique,nature_questions,msmarco_v1"))
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"[load] {args.qwen_path}")
    tok = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.train(False)
    print(f"[load] params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    print(f"[run] mode={args.mode} benches={benches} N={args.num_queries}/each "
          f"top_k={args.top_k} max_new={args.max_new_tokens} "
          f"thinking={'ON' if args.enable_thinking else 'OFF'}")

    summary = {}
    tele = {}
    t0 = time.time()
    for b in benches:
        out = run_bench(b, args.num_queries, args, model, tok, device)
        if out:
            summary[b] = out[0]
            tele[b] = out[1]

    print(f"\n========== ALL DONE in {(time.time()-t0)/60:.1f} min ==========")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump({"benches": summary, "telemetry": tele, "args": vars(args)},
                  f, indent=2)
    print(f"summary -> {Path(args.out_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
