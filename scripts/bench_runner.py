"""End-to-end bench runner for MSA-Qwen3.5-9B SFT-S2 inference.

Architecture:

  IR metric  ←  router top-k from ``SparseGenerator.retrieve``
                (operates in our offline-corpus index space; agg=mean)
  QA metric  ←  ``SparseGenerator.generate`` parsed via parse_msa_train_response
                (the LM emits training-time global gids — ignored — but the
                 answer string after <End-of-Retrieve> is reliable)

For each benchmark we:
  1. Subset the corpus to ``--num_docs`` documents.
  2. Pick eligible queries (gold refs all inside subset).
  3. For each query:
        - ``retrieve()`` → router-based pred_doc_ids (used for IR P/R/F1/IoU)
        - ``generate()`` → text → parse → pred_answer (used for QA EM/F1)
  4. Aggregate to a JSON summary per benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig

sys.path.append(str(Path(__file__).resolve().parent.parent))

from msa.model_msa_qwen3_5 import MSAQwen3_5Config, MSAQwen3_5ForCausalLM
from msa.lora_wrap import apply_lora_and_freeze, load_trainable_state_dict
from msa.inference import (
    SparseGenerator,
    load_benchmark,
    load_encoded_corpus,
    build_msa_train_prompt,
    parse_msa_train_response,
    best_qa_metrics,
    calculate_ir_metrics,
)
from msa.inference.offline_encoder import EncoderConfig, encode_corpus


QWEN3_5_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)


def build_model(qwen_path, ckpt_path, device):
    print(f"[setup] tokenizer + base config")
    tok = AutoTokenizer.from_pretrained(qwen_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base_cfg = AutoConfig.from_pretrained(qwen_path)
    text_kwargs = base_cfg.text_config.to_dict()
    text_kwargs.pop("model_type", None)
    for k in ("msa_layer_indices", "msa_chunk_size", "msa_top_k", "msa_aux_tau",
              "msa_aux_coef", "msa_num_router_heads", "msa_router_k_uses_kv_heads",
              "msa_max_docs", "msa_dropout", "msa_gradient_checkpointing"):
        text_kwargs.pop(k, None)
    cfg = MSAQwen3_5Config(
        **text_kwargs,
        msa_chunk_size=32, msa_top_k=8,
        msa_aux_tau=0.07, msa_aux_coef=1.0,
        msa_max_docs=64, msa_dropout=0.0,
        msa_num_router_heads=8,
        msa_router_k_uses_kv_heads=True,
    )
    model = MSAQwen3_5ForCausalLM(cfg)
    print(f"[setup] loading Qwen3.5 backbone (bf16)")
    rep = model.load_qwen3_5_pretrained(qwen_path, dtype=torch.bfloat16)
    print(f"        loaded={rep['loaded_count']} skipped={rep['skipped_from_src_count']}")
    apply_lora_and_freeze(
        model, r=16, alpha=32, dropout=0.0, targets=QWEN3_5_LORA_TARGETS,
    )
    if ckpt_path and str(ckpt_path).upper() not in ("RANDOM", "NONE"):
        print(f"[setup] loading SFT ckpt: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        load_trainable_state_dict(model, sd)
    else:
        print(f"[setup] RANDOM mode (ckpt={ckpt_path}): no trainable state loaded; LoRA + router stay random init")
    model = model.to(device).bfloat16()
    model.train(False)
    return model, tok


def run_one_bench(
    bench_name: str,
    sg: SparseGenerator,
    bench_data,
    eligible: list,
    args,
) -> dict:
    ir_sums = {k: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
               for k in args.ks}
    qa_sums = {"em": 0.0, "f1": 0.0}
    n = 0
    retrieve_time = 0.0
    generate_time = 0.0
    samples_log = []

    for qi, (sample, gold_idx) in enumerate(eligible):
        # --- IR: router-based retrieval ---
        r = sg.retrieve(query_text=sample.query,
                        k_docs=max(args.ks), agg=args.agg)
        retrieve_time += r.elapsed_sec
        for k in args.ks:
            preds = r.pred_doc_ids[:k]
            m = calculate_ir_metrics(gold_idx, preds)
            for key in ir_sums[k]:
                ir_sums[k][key] += m[key]

        # --- QA: full autoregressive generation ---
        if args.run_qa:
            prompt = build_msa_train_prompt(sample.query)
            g = sg.generate(
                prompt_text=prompt,
                max_new_tokens=args.max_new_tokens,
                stop_strings=["<|im_end|>"],
                temperature=0.0,
                use_kv_cache=args.use_kv_cache,
            )
            generate_time += g["elapsed_sec"]
            _, answer, dbg = parse_msa_train_response(g["text"])
            qa = best_qa_metrics(answer, [sample.answer])
            qa_sums["em"] += qa["em"]
            qa_sums["f1"] += qa["f1"]
            samples_log.append({
                "qi": qi, "query": sample.query[:120],
                "gold_answer": sample.answer[:120],
                "pred_answer": answer[:120],
                "stopped_by": g["stopped_by"],
                "n_new_tokens": g["n_new_tokens"],
                "saw_eor": dbg["saw_eor"], "saw_im_end": dbg["saw_im_end"],
                "router_top4": r.pred_doc_ids[:4],
                "gold_idx": gold_idx,
            })
        else:
            samples_log.append({
                "qi": qi, "query": sample.query[:120],
                "gold_answer": sample.answer[:120],
                "router_top4": r.pred_doc_ids[:4],
                "gold_idx": gold_idx,
            })
        n += 1

        if (qi + 1) % args.log_every == 0 or qi + 1 == len(eligible):
            kk = args.ks[-1]
            ir_p = ir_sums[kk]["precision"] / n
            ir_r = ir_sums[kk]["recall"] / n
            qa_em = qa_sums["em"] / n if args.run_qa else 0
            qa_f1 = qa_sums["f1"] / n if args.run_qa else 0
            print(f"  [{bench_name} {qi+1}/{len(eligible)}] "
                  f"ret_t={retrieve_time/n*1000:.0f}ms  gen_t={generate_time/n:.1f}s  "
                  f"P@{kk}={ir_p:.3f} R@{kk}={ir_r:.3f}  "
                  f"EM={qa_em:.3f}  textF1={qa_f1:.3f}")

    summary = {
        "bench": bench_name,
        "num_queries": n,
        "agg": args.agg,
        "num_docs_subset": args.num_docs,
        "ir": {k: {kk: vv / n for kk, vv in ir_sums[k].items()} for k in args.ks},
        "qa": ({"em": qa_sums["em"] / n, "f1": qa_sums["f1"] / n}
               if args.run_qa else None),
        "avg_retrieve_ms": retrieve_time / n * 1000,
        "avg_generate_sec": generate_time / n if args.run_qa else 0.0,
        "samples": samples_log,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen3_5_path", default="/workspace/qwen35_base")
    ap.add_argument("--ckpt", default="/workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt")
    ap.add_argument("--bench_root", default="/workspace/msa_bench")
    ap.add_argument("--benches", default="hotpotqa")  # comma list
    ap.add_argument("--num_docs", type=int, default=256)
    ap.add_argument("--num_queries", type=int, default=16)
    ap.add_argument("--max_doc_len", type=int, default=128)
    ap.add_argument("--encode_batch_size", type=int, default=16)
    ap.add_argument("--encoded_root",
                    default="/workspace/runs/sft_s2_qwen3_5_0426_0349/encoded")
    ap.add_argument("--corpus_suffix", default="",
                    help="Suffix appended to encoded dir name (e.g. _int4) for "
                         "loading a pre-quantized version of the same corpus.")
    ap.add_argument("--ks", type=lambda s: [int(x) for x in s.split(",")],
                    default=[1, 2, 4, 8, 16])
    ap.add_argument("--agg", default="mean", choices=["last", "mean", "max"])
    ap.add_argument("--run_qa", action="store_true",
                    help="run autoregressive generation for EM/F1 (slow)")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--log_every", type=int, default=4)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--use_kv_cache", action="store_true", default=True,
                    help="Use streaming KV-cached generate (10x faster). "
                         "Disable with --no_kv_cache for legacy slow path.")
    ap.add_argument("--no_kv_cache", dest="use_kv_cache", action="store_false")
    args = ap.parse_args()

    device = torch.device("cuda")
    model, tok = build_model(args.qwen3_5_path, args.ckpt, device)

    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    print(f"\n[runner] benches={benches}  num_docs={args.num_docs}  "
          f"num_queries={args.num_queries}  agg={args.agg}  run_qa={args.run_qa}")
    all_summaries = {}
    t_total = time.time()

    for bench_name in benches:
        print(f"\n========== {bench_name} ==========")
        bench_data = load_benchmark(bench_name, root=args.bench_root)
        docs = bench_data.documents[: args.num_docs]
        print(f"  subset {len(docs)} / full {bench_data.num_documents()}")

        encoded_dir = Path(args.encoded_root) / f"{bench_name}_{args.num_docs}{args.corpus_suffix}"
        if not (encoded_dir / "meta.json").exists():
            print(f"  [encode] -> {encoded_dir}")
            cfg = EncoderConfig(max_doc_len=args.max_doc_len,
                                batch_size=args.encode_batch_size,
                                dtype="bf16", device="cuda")
            encode_corpus(docs, model, tok, cfg, encoded_dir)
        enc = load_encoded_corpus(str(encoded_dir), device=device, dtype=torch.bfloat16)

        eligible = []
        for s in bench_data.samples:
            try:
                gold_idx = bench_data.label_indices(s)
            except KeyError:
                continue
            if all(0 <= g < args.num_docs for g in gold_idx):
                eligible.append((s, gold_idx))
            if len(eligible) >= args.num_queries:
                break
        print(f"  eligible queries: {len(eligible)} / {args.num_queries}")
        if not eligible:
            print(f"  WARN: no eligible queries — skip")
            continue

        sg = SparseGenerator(model=model, tokenizer=tok,
                             encoded_corpus=enc, device=device)
        all_summaries[bench_name] = run_one_bench(
            bench_name, sg, bench_data, eligible, args,
        )
        del enc, sg
        torch.cuda.empty_cache()

    elapsed = time.time() - t_total
    print(f"\n========== DONE in {elapsed/60:.1f} min ==========")
    for bench_name, s in all_summaries.items():
        kk = max(args.ks)
        print(f"\n{bench_name} (n={s['num_queries']}):")
        for k in args.ks:
            m = s["ir"][k]
            print(f"  IR  k={k:2d}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}  IoU={m['iou']:.3f}")
        if s["qa"] is not None:
            print(f"  QA  EM={s['qa']['em']:.3f}  textF1={s['qa']['f1']:.3f}")
        print(f"  latency: retrieve={s['avg_retrieve_ms']:.0f} ms  "
              f"generate={s['avg_generate_sec']:.1f} s")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"benches": all_summaries, "elapsed_min": elapsed/60}, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
