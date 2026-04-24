"""
Scaling-curve evaluator for MSA CPT (Paper §5.2 / Fig. 1 analog, scaled down).

For each memory-bank size N ∈ {8,16,32,64,128,256,512,1024,...}:
  build a held-out slice from MS MARCO v2.1 validation with `num_docs=N`
  (positives + N-|P| random negatives per sample)
  run three models and report:
    * router top-1 / top-4 / top-16 hit-a-positive
    * aux InfoNCE loss (lower = router more discriminative)
    * LM loss on the Generative-Retrieval target (lower = LM head emits right ids)

Models compared:
  A. MSA post-CPT  — our 4h CPT checkpoint
  B. MSA pre-CPT   — same architecture, backbone loaded from MiniMind,
                     router projectors random init (isolates CPT's benefit)
  C. MSA untied    — MiniMind backbone but MSA attention never trained
                     (optional — identical to B in our setup; kept as a hook)

Per-sample token count = N · L_d + L_q, so sweeping N sweeps the effective
"context length" the model perceives.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from msa.model_msa import MSAConfig, MSAForCausalLM
from msa.dataset_msa import build_msmarco_dataset, collate_msa
from msa.train_msa_cpt import _infer_config_from_minimind_state


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_model(
    over: dict,
    max_pos: int,
    top_k: int,
    minimind_sd: dict,
    trained_ckpt: str | None,
) -> MSAForCausalLM:
    cfg = MSAConfig(
        **over,
        msa_start_layer=4, msa_chunk_size=64, msa_top_k=top_k,
        max_position_embeddings=max_pos,
    )
    m = MSAForCausalLM(cfg).cuda().bfloat16()
    if trained_ckpt:
        sd = torch.load(trained_ckpt, map_location="cuda", weights_only=True)
        m.load_state_dict(sd, strict=False)
    else:
        m.load_minimind_pretrained({k: v.float() for k, v in minimind_sd.items()})
        m = m.bfloat16()
    m.eval()
    return m


@torch.no_grad()
def evaluate_model(model: MSAForCausalLM, loader: DataLoader, topk_values=(1, 4, 16)):
    """Returns dict of aggregate metrics over the loader."""
    hits = {k: 0 for k in topk_values}
    tot = 0
    aux_sum = 0.0
    lm_sum = 0.0
    lm_count = 0
    for batch in loader:
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        out = model(
            doc_input_ids=batch["doc_input_ids"],
            query_input_ids=batch["query_input_ids"],
            doc_attention_mask=batch["doc_attention_mask"],
            query_attention_mask=batch["query_attention_mask"],
            pos_doc_labels=batch["pos_doc_labels"],
            labels=batch["labels"],
            lm_loss_coef=1.0, aux_loss_coef=1.0,
        )
        s = out.routing_scores              # [B, N]
        pos = batch["pos_doc_labels"]        # [B, N]
        for k in topk_values:
            k_real = min(k, s.shape[1])
            idx = s.topk(k_real, dim=-1).indices
            hit = pos.gather(-1, idx).max(dim=-1).values    # [B]
            hits[k] += hit.sum().item()
        tot += pos.shape[0]
        aux_sum += float(out.aux_loss.item()) * pos.shape[0]
        if out.lm_loss is not None:
            lm_sum += float(out.lm_loss.item()) * pos.shape[0]
            lm_count += pos.shape[0]
    return {
        "tot": tot,
        "top1": hits.get(1, 0) / tot,
        "top4": hits.get(4, 0) / tot,
        "top16": hits.get(16, 0) / tot,
        "aux": aux_sum / tot,
        "lm": (lm_sum / lm_count) if lm_count else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpt_ckpt", default="../out/msa_cpt_paper.pth")
    ap.add_argument("--minimind_ckpt", default="../minimind/out/pretrain_768.pth")
    ap.add_argument("--tokenizer_path", default="../minimind/model")
    ap.add_argument("--queries", type=int, default=200,
                    help="number of held-out queries to use at each N")
    ap.add_argument("--n_list", default="8,16,32,64,128,256,512,1024",
                    help="comma-separated list of memory-bank sizes N")
    ap.add_argument("--max_doc_len", type=int, default=256)
    ap.add_argument("--max_query_len", type=int, default=256)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--out_json", default="../out/scaling_curve.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    mm_sd = torch.load(args.minimind_ckpt, map_location="cpu", weights_only=True)
    over = _infer_config_from_minimind_state(mm_sd)

    n_list = [int(x) for x in args.n_list.split(",")]
    log(f"N sweep: {n_list}")
    log(f"Per-sample context = N · {args.max_doc_len} + {args.max_query_len}")
    log(f"Effective context at max N: {n_list[-1] * args.max_doc_len + args.max_query_len} tokens")

    # Max position embeddings must be large enough for the biggest N (query offset = top_k,
    # doc-wise RoPE needs L_d slots, nothing depends on total N for RoPE).
    max_pos = max(args.max_doc_len, args.top_k + args.max_query_len) + 64
    log(f"max_position_embeddings = {max_pos}")

    results = {"n_list": n_list, "model_meta": {
        "backbone": "MiniMind-3 (64M dense, hidden=768, L=8, H=8, H_kv=4)",
        "cpt_ckpt": args.cpt_ckpt,
        "max_doc_len": args.max_doc_len,
        "max_query_len": args.max_query_len,
        "top_k": args.top_k,
    }}

    # Build datasets once per N (they share the held-out MS MARCO v2.1 validation corpus).
    model_variants = {
        "pre_cpt":  None,                 # fresh router, backbone from MiniMind
        "post_cpt": args.cpt_ckpt,        # our 4h run checkpoint
    }
    for variant, ckpt in model_variants.items():
        log(f"\n========== {variant} ==========")
        model = make_model(over, max_pos, args.top_k, mm_sd, ckpt)
        variant_results = {}
        for N in n_list:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ds = build_msmarco_dataset(
                tok, split="validation", version="v2.1",
                max_queries=args.queries, num_docs=N,
                max_doc_len=args.max_doc_len, max_query_len=args.max_query_len,
                seed=args.seed,
            )
            # The dataset's __len__ is the number of queries with positives; cap it.
            actual_queries = min(len(ds), args.queries)
            # Reduce batch when N is large to stay within VRAM.
            bs = max(1, args.batch_size)
            if N > 256: bs = max(1, bs // 2)
            if N > 512: bs = max(1, bs // 2)
            if N > 1024: bs = 1
            loader = DataLoader(ds, batch_size=bs, collate_fn=collate_msa, num_workers=2)
            t0 = time.time()
            m = evaluate_model(model, loader)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            ctx_tokens = N * args.max_doc_len + args.max_query_len
            log(f"N={N:5d}  ctx={ctx_tokens//1024}K tok  queries={actual_queries}  bs={bs}  "
                f"top1={m['top1']:.3f}  top4={m['top4']:.3f}  top16={m['top16']:.3f}  "
                f"aux={m['aux']:.3f}  lm={m['lm']:.3f}  peak={peak:.1f}GB  t={dt:.1f}s")
            variant_results[N] = {**m, "ctx_tokens": ctx_tokens, "peak_gb": peak, "time_s": dt}
        results[variant] = variant_results
        del model
        torch.cuda.empty_cache()

    # Write JSON.
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nWrote scaling curve → {out_path}")

    # Print a human-readable summary table.
    print()
    print("SCALING CURVE SUMMARY (router top-1 hit-a-positive, %)")
    header = f"{'N':>6}  {'ctx':>7}  " + "  ".join(f"{v:>10}" for v in model_variants)
    print(header)
    print("-" * len(header))
    for N in n_list:
        row = [f"{N:>6}", f"{N * args.max_doc_len // 1024:>5}K "]
        for v in model_variants:
            val = results[v][N]["top1"] * 100
            row.append(f"{val:>9.2f}%")
        print("  ".join(row))


if __name__ == "__main__":
    main()
