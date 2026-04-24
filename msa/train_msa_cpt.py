"""
MSA Continual Pre-training trainer.

Implements the two-phase optimization schedule from the paper (Sec. 3.3.1):

  Phase 1 (warmup)   : L = 0.1 * L_LLM + L_aux,   lr = 1e-4
  Phase 2 (main CPT) : L = 1.0 * L_LLM + 0.1 * L_aux, lr = 6e-6

During warmup the router projectors are aligned (aux loss dominates). During the main
phase, the Generative Retrieval objective becomes primary while aux keeps the routing
discriminative.

Default data source is the purely synthetic fact corpus, so the script runs end-to-end
with no external downloads. Pass --data t2t_mini to use MiniMind's pretrain_t2t_mini.jsonl.
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Make sibling package importable both as module and as script.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from msa.model_msa import MSAConfig, MSAForCausalLM
from msa.dataset_msa import (
    MSACPTDataset,
    build_synthetic_dataset,
    build_from_t2t_mini,
    build_msmarco_dataset,
    collate_msa,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_tokenizer(tokenizer_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_path)


def _infer_config_from_minimind_state(sd: dict) -> dict:
    """Infer hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
    intermediate_size, and vocab_size from a MiniMind checkpoint state_dict."""
    hidden_size = sd["model.embed_tokens.weight"].shape[1]
    vocab_size = sd["model.embed_tokens.weight"].shape[0]
    # q_proj: [H*D, hidden_size]; k_proj: [H_kv*D, hidden_size]; q_norm: [D]
    q_w = sd["model.layers.0.self_attn.q_proj.weight"].shape
    k_w = sd["model.layers.0.self_attn.k_proj.weight"].shape
    head_dim = sd["model.layers.0.self_attn.q_norm.weight"].shape[0]
    num_attention_heads = q_w[0] // head_dim
    num_key_value_heads = k_w[0] // head_dim
    inter = sd["model.layers.0.mlp.gate_proj.weight"].shape[0]
    # Count layers.
    layer_ids = set()
    for k in sd:
        if k.startswith("model.layers."):
            layer_ids.add(int(k.split(".")[2]))
    num_hidden_layers = max(layer_ids) + 1
    return dict(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=inter,
        vocab_size=vocab_size,
        head_dim=head_dim,
    )


def build_model(args, tokenizer) -> MSAForCausalLM:
    overrides = {}
    minimind_sd = None
    if args.from_minimind_weight and args.from_minimind_weight.lower() != "none":
        minimind_sd = torch.load(args.from_minimind_weight, map_location="cpu", weights_only=True)
        if isinstance(minimind_sd, dict) and "model" in minimind_sd and isinstance(minimind_sd["model"], dict):
            minimind_sd = minimind_sd["model"]
        overrides = _infer_config_from_minimind_state(minimind_sd)
        log(f"Inferred config from {args.from_minimind_weight}: {overrides}")

    cfg_kwargs = dict(
        hidden_size=overrides.get("hidden_size", args.hidden_size),
        num_hidden_layers=overrides.get("num_hidden_layers", args.num_hidden_layers),
        num_attention_heads=overrides.get("num_attention_heads", args.num_attention_heads),
        num_key_value_heads=overrides.get("num_key_value_heads", args.num_key_value_heads),
        vocab_size=overrides.get("vocab_size", tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)),
        max_position_embeddings=args.max_position_embeddings,
        msa_start_layer=args.msa_start_layer if args.msa_start_layer >= 0 else None,
        msa_chunk_size=args.msa_chunk_size,
        msa_top_k=args.msa_top_k,
        msa_aux_tau=args.aux_tau,
        msa_aux_coef=1.0,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2,
        tie_word_embeddings=True,
    )
    if "intermediate_size" in overrides:
        cfg_kwargs["intermediate_size"] = overrides["intermediate_size"]
    if "head_dim" in overrides:
        cfg_kwargs["head_dim"] = overrides["head_dim"]

    cfg = MSAConfig(**cfg_kwargs)
    model = MSAForCausalLM(cfg)
    if minimind_sd is not None:
        info = model.load_minimind_pretrained(minimind_sd, strict=False)
        log(f"Loaded {len(info['loaded'])} tensors from MiniMind checkpoint")
    return model


def build_dataset(args, tokenizer) -> MSACPTDataset:
    if args.data == "synthetic":
        return build_synthetic_dataset(
            tokenizer, n_facts=args.num_facts,
            num_docs=args.num_docs, max_doc_len=args.max_doc_len, max_query_len=args.max_query_len,
            seed=args.seed,
        )
    if args.data == "t2t_mini":
        return build_from_t2t_mini(
            tokenizer, jsonl_path=args.t2t_mini_path, max_facts=args.num_facts,
            num_docs=args.num_docs, max_doc_len=args.max_doc_len, max_query_len=args.max_query_len,
            seed=args.seed,
        )
    if args.data == "ms_marco":
        return build_msmarco_dataset(
            tokenizer, split=args.msmarco_split, version=args.msmarco_version,
            max_queries=args.num_facts,
            num_docs=args.num_docs, max_doc_len=args.max_doc_len, max_query_len=args.max_query_len,
            seed=args.seed,
        )
    raise ValueError(args.data)


def get_lr(step, total_steps, base_lr, min_ratio: float = 0.1):
    """Cosine schedule shared with MiniMind's get_lr."""
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1))))


def phase_coefs(phase: str) -> tuple[float, float]:
    """(lm_coef, aux_coef) per Paper Sec. 3.3.1."""
    if phase == "warmup":
        return 0.1, 1.0
    elif phase == "main":
        return 1.0, 0.1
    raise ValueError(phase)


def run_phase(
    phase: str,
    model: MSAForCausalLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    args,
    total_steps: int,
    base_lr: float,
    device: str,
    dtype: torch.dtype,
    autocast_ctx,
    tokens_per_sample: int,
    run_state: dict,
):
    """Runs one phase (warmup or main).

    ``run_state`` threads cumulative state across phases:
        tokens_seen (int), global_step (int), wall_start (float), ckpt_dir (str).
    """
    lm_coef, aux_coef = phase_coefs(phase)
    log(f"=== Phase: {phase} | lm_coef={lm_coef} aux_coef={aux_coef} base_lr={base_lr} total_steps={total_steps} ===")
    model.train()
    step = 0
    running = dict(loss=0.0, lm=0.0, aux=0.0, count=0)
    target_tokens = 158_950_000_000  # Paper's CPT corpus (Sec. 3.3.1)
    ckpt_dir = run_state["ckpt_dir"]

    for epoch in range(args.epochs):
        for batch in loader:
            # Wall-clock guard.
            elapsed = time.time() - run_state["wall_start"]
            if args.max_train_seconds > 0 and elapsed >= args.max_train_seconds:
                log(f"[{phase}] stopping: hit max_train_seconds={args.max_train_seconds}")
                return
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            lr = get_lr(step, total_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                out = model(
                    doc_input_ids=batch["doc_input_ids"],
                    query_input_ids=batch["query_input_ids"],
                    doc_attention_mask=batch["doc_attention_mask"],
                    query_attention_mask=batch["query_attention_mask"],
                    pos_doc_labels=batch["pos_doc_labels"],
                    labels=batch["labels"],
                    lm_loss_coef=lm_coef,
                    aux_loss_coef=aux_coef,
                )
            loss = out.loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            running["loss"] += float(out.loss.detach())
            running["lm"] += float(out.lm_loss.detach()) if out.lm_loss is not None else 0.0
            running["aux"] += float(out.aux_loss.detach()) if out.aux_loss is not None else 0.0
            running["count"] += 1

            step += 1
            run_state["global_step"] += 1
            run_state["tokens_seen"] += batch["doc_input_ids"].shape[0] * tokens_per_sample

            if step % args.log_interval == 0 or step == total_steps:
                n = max(running["count"], 1)
                toks = run_state["tokens_seen"]
                pct = 100.0 * toks / target_tokens
                log(
                    f"[{phase}] step {step}/{total_steps} "
                    f"loss={running['loss']/n:.4f} lm={running['lm']/n:.4f} aux={running['aux']/n:.4f} "
                    f"lr={lr:.2e} tokens={_human_tokens(toks)} ({pct:.3f}% of 158.95B) "
                    f"elapsed={elapsed/60:.1f}min"
                )
                running = dict(loss=0.0, lm=0.0, aux=0.0, count=0)

            if args.ckpt_every > 0 and run_state["global_step"] % args.ckpt_every == 0:
                path = os.path.join(ckpt_dir, f"{args.save_name}_step{run_state['global_step']}.pth")
                sd = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
                torch.save(sd, path)
                log(f"[{phase}] saved periodic checkpoint {path}")

            if step >= total_steps:
                return


def _human_tokens(n: int) -> str:
    for u, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{u}"
    return str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "t2t_mini", "ms_marco"], default="ms_marco")
    ap.add_argument("--t2t_mini_path", default="../minimind/dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--msmarco_split", default="train")
    ap.add_argument("--msmarco_version", default="v2.1", choices=["v1.1", "v2.1"],
                    help="v2.1 is ~10x larger (808K vs 82K queries)")
    ap.add_argument("--num_facts", type=int, default=0, help="0 = use full split (MS MARCO: queries with labelled positives)")
    ap.add_argument("--num_docs", type=int, default=32, help="docs per training sample; must be >> top_k")
    ap.add_argument("--max_doc_len", type=int, default=96)
    ap.add_argument("--max_query_len", type=int, default=160)

    ap.add_argument("--tokenizer_path", default="../minimind/model")
    ap.add_argument("--from_minimind_weight", default="none", help="Path to a MiniMind .pth to warm-start backbone.")

    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_hidden_layers", type=int, default=4)
    ap.add_argument("--num_attention_heads", type=int, default=4)
    ap.add_argument("--num_key_value_heads", type=int, default=2)
    ap.add_argument("--max_position_embeddings", type=int, default=2048)
    ap.add_argument("--msa_start_layer", type=int, default=-1, help="-1 = num_hidden_layers // 2 (later half).")
    ap.add_argument("--msa_chunk_size", type=int, default=32)
    ap.add_argument("--msa_top_k", type=int, default=4, help="top-k docs selected; num_docs should be >> top_k")
    ap.add_argument("--aux_tau", type=float, default=0.07)

    ap.add_argument("--assert_sparse", action="store_true", help="abort if num_docs <= msa_top_k (no real sparsity)")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10, help="loops over the loader until total_steps hit")
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--main_steps", type=int, default=500)
    ap.add_argument("--warmup_lr", type=float, default=1e-4)
    ap.add_argument("--main_lr", type=float, default=6e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=0, help="save a checkpoint every N global steps (0=off)")
    ap.add_argument("--max_train_seconds", type=int, default=0, help="hard wall-clock cap; 0=no cap")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default="../out")
    ap.add_argument("--save_name", default="msa_cpt")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    if args.num_docs <= args.msa_top_k:
        msg = (f"WARNING: num_docs={args.num_docs} <= msa_top_k={args.msa_top_k}; "
               f"top-k selection is degenerate (all docs always selected).")
        if args.assert_sparse:
            raise SystemExit(msg)
        log(msg)

    tokenizer = build_tokenizer(args.tokenizer_path)
    model = build_model(args, tokenizer).to(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Model params: {n_params:.2f}M")

    dataset = build_dataset(args, tokenizer)
    log(f"Dataset size: {len(dataset)}; num_docs/sample={args.num_docs}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_msa,
        pin_memory=(args.device.startswith("cuda")),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.warmup_lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Mixed precision.
    if args.dtype == "bf16" and args.device.startswith("cuda"):
        dtype = torch.bfloat16
        scaler = None
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    elif args.dtype == "fp16" and args.device.startswith("cuda"):
        dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        dtype = torch.float32
        scaler = None
        autocast_ctx = nullcontext()

    # Shared run state for token-counting and wall-clock guards.
    tokens_per_sample = args.num_docs * args.max_doc_len + args.max_query_len
    run_state = {
        "tokens_seen": 0,
        "global_step": 0,
        "wall_start": time.time(),
        "ckpt_dir": args.save_dir,
    }
    log(f"Tokens per sample: {tokens_per_sample}  (num_docs*max_doc_len + max_query_len)")
    log(f"Target corpus: 158.95B tokens  (paper Sec. 3.3.1)")

    # ---- Phase 1: router-warmup
    run_phase(
        "warmup", model, loader, optimizer, scaler, args,
        total_steps=args.warmup_steps, base_lr=args.warmup_lr,
        device=args.device, dtype=dtype, autocast_ctx=autocast_ctx,
        tokens_per_sample=tokens_per_sample, run_state=run_state,
    )

    # ---- Phase 2: main CPT
    run_phase(
        "main", model, loader, optimizer, scaler, args,
        total_steps=args.main_steps, base_lr=args.main_lr,
        device=args.device, dtype=dtype, autocast_ctx=autocast_ctx,
        tokens_per_sample=tokens_per_sample, run_state=run_state,
    )

    out_path = os.path.join(args.save_dir, f"{args.save_name}.pth")
    sd = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, out_path)
    log(f"Saved {out_path}")
    log(f"Final: tokens_seen={_human_tokens(run_state['tokens_seen'])}  "
        f"({100*run_state['tokens_seen']/158_950_000_000:.3f}% of paper's 158.95B)")


if __name__ == "__main__":
    main()
