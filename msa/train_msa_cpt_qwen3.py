"""MSA CPT on top of Qwen3-8B-Base with LoRA backbone + fully-trainable router.

Changes compared to ``train_msa_cpt.py``:

* Backbone switched from MiniMind to Qwen3 via :mod:`msa.model_msa_qwen3`.
* Qwen3 dense linears are wrapped in LoRA adapters (r, alpha configurable);
  everything else in the backbone (RMSNorms, embeddings, lm_head) is frozen.
* Router projectors (``qr_proj`` / ``kr_proj``) are fully trainable with a
  dedicated learning rate, typically larger than the LoRA LR.
* Optimizer uses param groups: one for LoRA (wd=0, lr=lora_lr), one for
  router (wd=0.01, lr=router_lr). Each group gets its own cosine schedule.
* Two-phase schedule (warmup/main) from the paper is preserved; the only
  change is that the LR under the phase gets split into ``lora_lr`` and
  ``router_lr`` scaled proportionally.

The data pipeline (``dataset_msa.py``) is reused as-is. Qwen3 tokenizer
(ChatML-style) is fully compatible with William's prompt templates:
``<|im_start|>``, ``<|im_end|>``, ``<|object_ref_end|>`` all exist in
Qwen3 vocab as added tokens.
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen3Config

sys.path.append(str(Path(__file__).resolve().parent.parent))
from msa.model_msa_qwen3 import MSAQwen3Config, MSAQwen3ForCausalLM
from msa.lora_wrap import (
    apply_lora_and_freeze,
    split_params_for_optim,
    trainable_report,
    trainable_state_dict,
)
from msa.dataset_msa import (
    build_synthetic_dataset,
    build_from_t2t_mini,
    build_msmarco_dataset,
    collate_msa,
)
from msa.dataset_msa_sft import build_sft_mix
from msa.train_utils import (
    WandbLogger,
    RedLineMonitor,
    RedLineTriggered,
    compute_router_precision_on_batch,
    mini_eval_llm_judge,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------
def build_tokenizer(qwen3_path: str):
    return AutoTokenizer.from_pretrained(qwen3_path, trust_remote_code=False)


def build_model(args) -> tuple[MSAQwen3ForCausalLM, dict]:
    """Build :class:`MSAQwen3ForCausalLM` from a Qwen3 HF checkpoint and inject LoRA."""
    qwen3_cfg = Qwen3Config.from_pretrained(args.qwen3_path)
    base_kwargs = qwen3_cfg.to_dict()
    # Guard against conflicts with explicit MSA kwargs.
    for k in ("msa_start_layer", "msa_chunk_size", "msa_top_k", "msa_aux_tau",
              "msa_aux_coef", "msa_num_router_heads", "msa_router_k_uses_kv_heads",
              "msa_max_docs", "msa_dropout"):
        base_kwargs.pop(k, None)

    cfg = MSAQwen3Config(
        **base_kwargs,
        msa_start_layer=args.msa_start_layer if args.msa_start_layer >= 0 else None,
        msa_chunk_size=args.msa_chunk_size,
        msa_top_k=args.msa_top_k,
        msa_aux_tau=args.aux_tau,
        msa_aux_coef=1.0,
        msa_max_docs=args.msa_max_docs,
        msa_dropout=args.msa_dropout,
        msa_num_router_heads=args.num_router_heads if args.num_router_heads > 0 else None,
        msa_router_k_uses_kv_heads=bool(args.router_k_uses_kv_heads),
    )
    log(f"MSA layer span: [{cfg.msa_start_layer}, {cfg.num_hidden_layers}) "
        f"of {cfg.num_hidden_layers} total layers")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    # Build on CPU first to avoid OOM during init on single 80GB GPU.
    model = MSAQwen3ForCausalLM(cfg)
    log(f"Loading Qwen3 backbone from {args.qwen3_path} (dtype={args.dtype}) ...")
    info = model.load_qwen3_pretrained(args.qwen3_path, dtype=dtype)
    log(f"Loaded {len(info['loaded'])} tensors; {len(info['missing_in_src'])} fresh "
        f"(router + any new params)")

    # Cast backbone to the chosen dtype before LoRA injection so LoRA A/B keep
    # their own (fp32) dtype for stability.
    for p in model.parameters():
        if p.dtype.is_floating_point:
            p.data = p.data.to(dtype)

    report = apply_lora_and_freeze(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    # LoRA adapters default to fp32 — keep that for stable gradients.
    log(
        "LoRA injected: r={r} alpha={a} dropout={d} wrapped={n} linears. "
        "Trainable: {tr:.2f}M / total {tot:.2f}M ({pct:.3f}%). "
        "router={r_cnt:.2f}M, lora={l_cnt:.2f}M".format(
            r=args.lora_r, a=args.lora_alpha, d=args.lora_dropout,
            n=report["n_lora_linears_wrapped"],
            tr=report["trainable"] / 1e6, tot=report["total"] / 1e6,
            pct=report["trainable_pct"], r_cnt=report["router"] / 1e6,
            l_cnt=report["lora"] / 1e6,
        )
    )
    return model, report


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataset(args, tokenizer):
    if args.data == "synthetic":
        return build_synthetic_dataset(
            tokenizer, n_facts=args.num_facts,
            num_docs=args.num_docs, max_doc_len=args.max_doc_len,
            max_query_len=args.max_query_len, seed=args.seed,
        )
    if args.data == "t2t_mini":
        return build_from_t2t_mini(
            tokenizer, jsonl_path=args.t2t_mini_path, max_facts=args.num_facts,
            num_docs=args.num_docs, max_doc_len=args.max_doc_len,
            max_query_len=args.max_query_len, seed=args.seed,
        )
    if args.data == "ms_marco":
        return build_msmarco_dataset(
            tokenizer, split=args.msmarco_split, version=args.msmarco_version,
            max_queries=args.num_facts,
            num_docs=args.num_docs, max_doc_len=args.max_doc_len,
            max_query_len=args.max_query_len, seed=args.seed,
        )
    if args.data == "sft_mix":
        sft_datasets = [s.strip() for s in (args.sft_datasets or "").split(",") if s.strip()] or None
        return build_sft_mix(
            tokenizer, datasets=sft_datasets,
            max_queries_per=args.num_facts if args.num_facts else 20000,
            num_docs=args.num_docs, max_doc_len=args.max_doc_len,
            max_query_len=args.max_query_len, seed=args.seed,
        )
    if args.data == "shard":
        from msa.dataset_msa_shard import load_shard
        return load_shard(
            args.shard_dir, tokenizer,
            num_docs=args.num_docs,
            max_doc_len=args.max_doc_len,
            max_query_len=args.max_query_len,
            seed=args.seed,
            include_doc_text_in_target=True,
            sample_limit=args.num_facts if args.num_facts else 0,
        )
    raise ValueError(args.data)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def cosine_lr(step: int, total_steps: int, base_lr: float, min_ratio: float = 0.1) -> float:
    return base_lr * (
        min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))
    )


def phase_coefs(phase: str) -> tuple[float, float]:
    if phase == "warmup":
        return 0.1, 1.0
    if phase == "main":
        return 1.0, 0.1
    raise ValueError(phase)


def run_phase(
    phase: str,
    model: MSAQwen3ForCausalLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    group_base_lrs: dict[int, float],
    scaler,
    args,
    total_steps: int,
    device: str,
    autocast_ctx,
    tokens_per_sample: int,
    run_state: dict,
):
    lm_coef, aux_coef = phase_coefs(phase)
    log(f"=== Phase: {phase} | lm_coef={lm_coef} aux_coef={aux_coef} total_steps={total_steps} ===")
    model.train()
    step = 0
    running = dict(loss=0.0, lm=0.0, aux=0.0, count=0)
    router_running = dict(precision=0.0, recall=0.0, count=0)
    target_tokens = int(args.target_tokens)
    ckpt_dir = run_state["ckpt_dir"]
    wandb_log: WandbLogger = run_state["wandb"]
    monitor: RedLineMonitor = run_state["monitor"]

    for epoch in range(args.epochs):
        for batch in loader:
            elapsed = time.time() - run_state["wall_start"]
            if args.max_train_seconds > 0 and elapsed >= args.max_train_seconds:
                log(f"[{phase}] stopping: hit max_train_seconds={args.max_train_seconds}")
                return
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Per-group cosine scheduling; each group has its own base LR.
            for gi, pg in enumerate(optimizer.param_groups):
                pg["lr"] = cosine_lr(step, total_steps, group_base_lrs[gi])

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
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), args.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), args.grad_clip
                )
                optimizer.step()

            running["loss"] += float(out.loss.detach())
            running["lm"] += float(out.lm_loss.detach()) if out.lm_loss is not None else 0.0
            running["aux"] += float(out.aux_loss.detach()) if out.aux_loss is not None else 0.0
            running["count"] += 1

            # Router precision / recall on the current batch (cheap; no extra fwd).
            if (run_state["global_step"] + 1) % args.router_eval_every == 0:
                routing_scores = getattr(out, "routing_scores", None)
                if routing_scores is not None:
                    rmetrics = compute_router_precision_on_batch(
                        routing_scores=routing_scores,
                        pos_doc_labels=batch["pos_doc_labels"],
                        top_k=args.msa_top_k,
                    )
                    router_running["precision"] += rmetrics["router_precision"]
                    router_running["recall"] += rmetrics["router_recall"]
                    router_running["count"] += 1

            step += 1
            run_state["global_step"] += 1
            run_state["tokens_seen"] += batch["doc_input_ids"].shape[0] * tokens_per_sample

            if step % args.log_interval == 0 or step == total_steps:
                n = max(running["count"], 1)
                toks = run_state["tokens_seen"]
                pct = 100.0 * toks / target_tokens
                lrs = "/".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
                rn = max(router_running["count"], 1)
                router_p = router_running["precision"] / rn if router_running["count"] else None
                router_r = router_running["recall"] / rn if router_running["count"] else None
                router_str = (
                    f" router_p={router_p:.3f} router_r={router_r:.3f}"
                    if router_p is not None else ""
                )
                log(
                    f"[{phase}] step {step}/{total_steps} "
                    f"loss={running['loss']/n:.4f} lm={running['lm']/n:.4f} "
                    f"aux={running['aux']/n:.4f}{router_str} lr=[{lrs}] "
                    f"tokens={_human_tokens(toks)} ({pct:.3f}%) "
                    f"elapsed={elapsed/60:.1f}min"
                )
                metrics = {
                    f"{phase}/loss": running["loss"] / n,
                    f"{phase}/lm_loss": running["lm"] / n,
                    f"{phase}/aux_loss": running["aux"] / n,
                    f"{phase}/lr_lora": optimizer.param_groups[0]["lr"],
                    f"{phase}/lr_router": optimizer.param_groups[1]["lr"],
                    "tokens_seen": toks,
                    "tokens_pct_of_target": pct,
                    "elapsed_min": elapsed / 60,
                    "loss": running["loss"] / n,
                }
                if router_p is not None:
                    metrics[f"{phase}/router_precision"] = router_p
                    metrics[f"{phase}/router_recall"] = router_r
                    metrics["router_precision"] = router_p
                    metrics["router_recall"] = router_r
                wandb_log.log(metrics, step=run_state["global_step"])
                monitor.update(metrics, step=run_state["global_step"])
                running = dict(loss=0.0, lm=0.0, aux=0.0, count=0)
                router_running = dict(precision=0.0, recall=0.0, count=0)

            # Mid-eval (slow, optional, requires paper-aligned encoded_corpora).
            if (
                args.mid_eval_every > 0
                and run_state["global_step"] % args.mid_eval_every == 0
                and step > 0
            ):
                log(f"[{phase}] mid-eval @ step {run_state['global_step']} ...")
                eval_t0 = time.time()
                eval_out = mini_eval_llm_judge(
                    model=model, tokenizer=run_state["tokenizer"],
                    encoded_root=Path(args.encoded_root),
                    bench_root=Path(args.bench_root),
                    bench_name=args.mid_eval_bench,
                    n_queries=args.mid_eval_n,
                    top_k=args.msa_top_k,
                    max_new_tokens=256, device=args.device,
                    judge_backend=args.mid_eval_judge,
                )
                eval_elapsed = time.time() - eval_t0
                log(
                    f"[{phase}] mid-eval done in {eval_elapsed:.1f}s: "
                    f"empty={eval_out.get('empty_rate'):.3f} "
                    f"router_f1={eval_out.get('router_f1'):.3f} "
                    f"judge={eval_out.get('judge_llm')}"
                    f"{' err=' + eval_out['error'] if eval_out.get('error') else ''}"
                )
                wandb_log.log({
                    f"eval/{args.mid_eval_bench}/empty_rate": eval_out.get("empty_rate"),
                    f"eval/{args.mid_eval_bench}/router_f1": eval_out.get("router_f1"),
                    f"eval/{args.mid_eval_bench}/judge_llm": eval_out.get("judge_llm"),
                    "empty_answer_rate": eval_out.get("empty_rate"),
                    "judge_llm": eval_out.get("judge_llm"),
                }, step=run_state["global_step"])
                monitor.update({
                    "empty_answer_rate": eval_out.get("empty_rate"),
                    "judge_llm": eval_out.get("judge_llm"),
                }, step=run_state["global_step"])
                model.train()

            if args.ckpt_every > 0 and run_state["global_step"] % args.ckpt_every == 0:
                path = os.path.join(ckpt_dir, f"{args.save_name}_step{run_state['global_step']}.pt")
                torch.save(trainable_state_dict(model), path)
                log(f"[{phase}] saved trainable ckpt {path}")

            if step >= total_steps:
                return


def _human_tokens(n: int) -> str:
    for u, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{u}"
    return str(n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--data", choices=["synthetic", "t2t_mini", "ms_marco", "sft_mix", "shard"], default="ms_marco")
    ap.add_argument("--shard_dir", type=str, default="", help="Path to shard_X dir (for --data shard)")
    ap.add_argument("--sft_datasets", type=str, default="hotpotqa,musique,triviaqa,nq,msmarco",
                    help="comma-list of bench names for sft_mix; default = William's 5-bench")
    ap.add_argument("--resume_ckpt", type=str, default="",
                    help="path to a CPT trainable .pt to resume from (loads after LoRA inject)")
    ap.add_argument("--t2t_mini_path", default="../minimind/dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--msmarco_split", default="train")
    ap.add_argument("--msmarco_version", default="v2.1", choices=["v1.1", "v2.1"])
    ap.add_argument("--num_facts", type=int, default=0, help="0 = full split")
    # v4-4B retrain: paper-aligned (was num_docs=32, max_doc_len=128).
    ap.add_argument("--num_docs", type=int, default=64)
    ap.add_argument("--max_doc_len", type=int, default=192)
    ap.add_argument("--max_query_len", type=int, default=256)

    # Backbone
    ap.add_argument("--qwen3_path", required=True, help="HF dir of Qwen3-*-Base")

    # MSA
    ap.add_argument("--msa_start_layer", type=int, default=-1, help="-1 = num_hidden_layers // 2")
    ap.add_argument("--msa_chunk_size", type=int, default=64)
    ap.add_argument("--msa_top_k", type=int, default=16)
    ap.add_argument("--msa_max_docs", type=int, default=64)
    ap.add_argument("--msa_dropout", type=float, default=0.0)
    ap.add_argument("--aux_tau", type=float, default=0.07)
    ap.add_argument("--assert_sparse", action="store_true")
    ap.add_argument("--num_router_heads", type=int, default=-1,
                    help="-1 = num_attention_heads (32 for Qwen3-8B); set to num_key_value_heads "
                         "(8) to align with GQA and cut router params by ~60%")
    ap.add_argument("--router_k_uses_kv_heads", type=int, default=1,
                    help="1 = router-K uses num_key_value_heads; 0 = use num_router_heads")

    # LoRA
    # v4-4B retrain: paper-aligned r=64/alpha=128 (William: 原版4b + lora).
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=float, default=128.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Optim
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--main_steps", type=int, default=1000)
    ap.add_argument("--warmup_lora_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_router_lr", type=float, default=5e-4)
    ap.add_argument("--main_lora_lr", type=float, default=5e-5)
    ap.add_argument("--main_router_lr", type=float, default=2e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=5)
    ap.add_argument("--ckpt_every", type=int, default=0)
    ap.add_argument("--max_train_seconds", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default="../out")
    ap.add_argument("--save_name", default="msa_qwen3_cpt")

    # ----- Monitoring (v4: never train blind) -----
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--router_eval_every", type=int, default=200)
    ap.add_argument("--mid_eval_every", type=int, default=0)
    ap.add_argument("--mid_eval_bench", default="hotpotqa")
    ap.add_argument("--mid_eval_n", type=int, default=30)
    ap.add_argument("--mid_eval_judge", default="skip", choices=["skip", "openrouter"])
    ap.add_argument("--bench_root", default="/workspace/msa_bench")
    ap.add_argument("--encoded_root", default="/workspace/encoded_corpora")
    ap.add_argument("--auto_kill", action="store_true")
    ap.add_argument("--target_tokens", type=float, default=158.95e9)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ----- Monitoring init -----
    wandb_log = WandbLogger(
        project=args.wandb_project,
        run_name=args.wandb_name or args.save_name,
        config=vars(args), group=args.wandb_group,
    )
    stop_marker = Path(args.save_dir) / f"{args.save_name}.kill_marker"
    monitor = RedLineMonitor(
        rules=None if args.auto_kill else [],
        stop_marker_path=stop_marker,
        wandb=wandb_log,
    )

    if args.num_docs <= args.msa_top_k:
        msg = (f"WARNING: num_docs={args.num_docs} <= msa_top_k={args.msa_top_k}; "
               f"top-k degenerate.")
        if args.assert_sparse:
            raise SystemExit(msg)
        log(msg)

    log(f"Tokenizer: {args.qwen3_path}")
    tokenizer = build_tokenizer(args.qwen3_path)
    model, _report = build_model(args)

    # Resume from a prior trainable state dict (e.g. CPT ckpt before SFT).
    # Loaded BEFORE .to(device) so we don't waste GPU mem on the brief CPU copy.
    if args.resume_ckpt:
        log(f"Resuming trainable state dict from {args.resume_ckpt}")
        sd = torch.load(args.resume_ckpt, map_location="cpu", weights_only=True)
        info = model.load_state_dict(sd, strict=False)
        log(f"  loaded {len(sd)} keys, missing={len(info.missing_keys)}, "
            f"unexpected={len(info.unexpected_keys)}")

    model.to(args.device)

    # Track trainable tensor counts per group after .to(device).
    lora_ps, router_ps, other_ps = split_params_for_optim(model)
    log(f"Optimizer groups — lora:{sum(p.numel() for p in lora_ps)/1e6:.2f}M, "
        f"router:{sum(p.numel() for p in router_ps)/1e6:.2f}M, "
        f"other:{sum(p.numel() for p in other_ps)/1e6:.2f}M")
    if other_ps:
        log(f"NOTE: {len(other_ps)} other trainable params present — they will not be optimized "
            f"(only LoRA+router groups are wired up).")

    dataset = build_dataset(args, tokenizer)
    log(f"Dataset size: {len(dataset)}; num_docs/sample={args.num_docs}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_msa,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )

    # Dtype/autocast setup
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

    # Initial LRs will be overwritten each step via cosine; we still set sane
    # starting values so the optimizer initializes correctly.
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_ps, "lr": args.warmup_lora_lr, "weight_decay": 0.0},
            {"params": router_ps, "lr": args.warmup_router_lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.95),
    )

    tokens_per_sample = args.num_docs * args.max_doc_len + args.max_query_len
    run_state = {
        "tokens_seen": 0,
        "global_step": 0,
        "wall_start": time.time(),
        "ckpt_dir": args.save_dir,
        "wandb": wandb_log,
        "monitor": monitor,
        "tokenizer": tokenizer,
    }
    log(f"Tokens per sample: {tokens_per_sample}")

    try:
        # Phase 1: router-warmup
        run_phase(
            "warmup", model, loader, optimizer,
            group_base_lrs={0: args.warmup_lora_lr, 1: args.warmup_router_lr},
            scaler=scaler, args=args, total_steps=args.warmup_steps,
            device=args.device, autocast_ctx=autocast_ctx,
            tokens_per_sample=tokens_per_sample, run_state=run_state,
        )
        # Phase 2: main CPT
        run_phase(
            "main", model, loader, optimizer,
            group_base_lrs={0: args.main_lora_lr, 1: args.main_router_lr},
            scaler=scaler, args=args, total_steps=args.main_steps,
            device=args.device, autocast_ctx=autocast_ctx,
            tokens_per_sample=tokens_per_sample, run_state=run_state,
        )
    except RedLineTriggered as e:
        log(f"[RED-LINE] training killed: {e}")
        kill_path = os.path.join(args.save_dir, f"{args.save_name}.killed.pt")
        torch.save(trainable_state_dict(model), kill_path)
        log(f"Saved trainable state dict to {kill_path}")
        wandb_log.finish()
        sys.exit(2)

    out_path = os.path.join(args.save_dir, f"{args.save_name}.pt")
    torch.save(trainable_state_dict(model), out_path)
    log(f"Saved trainable state dict to {out_path}")
    log(f"Final: tokens_seen={_human_tokens(run_state['tokens_seen'])}  "
        f"({100*run_state['tokens_seen']/158_950_000_000:.3f}% of paper's 158.95B)")


if __name__ == "__main__":
    main()
