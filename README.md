# MSA × MiniMind

MSA (Memory Sparse Attention, [arXiv:2603.23516](https://arxiv.org/abs/2603.23516))
continual pre-training implementation on top of the
[MiniMind-3](https://github.com/jingyaogong/minimind) 64 M backbone.

> Goal: a paper-faithful, single-GPU reproduction of MSA's end-to-end training
> stack — architecture, losses, data pipeline, and evaluation protocol — at a
> scale that actually runs on a single H200.

---

## What's in this repo

```
msa/
├── model_msa.py         Memory-Sparse-Attention layers, config, and MSAForCausalLM
├── dataset_msa.py       Generative-Retrieval CPT dataset; MS MARCO / synthetic / t2t_mini
├── train_msa_cpt.py     Two-phase CPT trainer (warmup → main)
├── benchmarks.py        Adapters for the 9 benchmarks in the paper (6 implemented)
└── eval_scaling.py      Scaling-curve evaluator (paper §5.2 / Fig. 1 analog)

SUMMARY_CN.md            Detailed step-by-step of the work (Chinese)
PLAN_CN.md               Completion plan — what's done + what remains (Chinese)
out/scaling_curve.json   Post-run scaling-curve metrics (pre-CPT vs post-CPT, N=8..1024)
MSA.pdf                  Paper (arXiv 2603.23516)
pyproject.toml, uv.lock  Pinned environment (Python 3.10+, torch 2.6 CU124)
```

Everything paper-specified is aligned:
chunk size P=64, top-k=16, doc-wise + global RoPE, Eq. (2) routing,
Eq. (5) supervised InfoNCE aux loss, two-phase LR schedule
`0.1·L_LLM + L_aux @ 1e-4` → `L_LLM + 0.1·L_aux @ 6e-6`,
Generative-Retrieval target format per Fig. 3 (global doc-IDs + original-text
injection + `<End-of-Retrieve>` + answer), GQA-aligned router-K matching
the official `EverMind-AI/MSA` reference.

---

## Quickstart

**1. Environment**
```bash
cd msa-minimind
uv sync              # installs torch-cu124, transformers, datasets, etc.
```

**2. Get the MiniMind backbone weights & CPT corpus**
```bash
# MiniMind-3 pretrained weights (132 MB, dense)
uv run modelscope download --model gongjy/minimind-3-pytorch \
  --local_dir minimind/out

# MS MARCO v2.1 is pulled automatically by the dataset builder on first use.
# Optional MiniMind t2t corpus (1.2 GB):
uv run modelscope download --dataset gongjy/minimind_dataset \
  pretrain_t2t_mini.jsonl --local_dir minimind/dataset
```

**3. Run CPT (paper-aligned)**
```bash
cd msa
uv run python train_msa_cpt.py \
  --data ms_marco --msmarco_version v2.1 \
  --from_minimind_weight ../minimind/out/pretrain_768.pth \
  --num_docs 64 --msa_top_k 16 --msa_chunk_size 64 --msa_start_layer 4 \
  --max_doc_len 256 --max_query_len 256 \
  --warmup_steps 2000 --main_steps 40000 \
  --batch_size 8 --ckpt_every 2000 \
  --max_train_seconds 14400 \
  --dtype bf16 --save_name msa_cpt_paper
```

**4. Scaling-curve evaluation**
```bash
cd msa
uv run python eval_scaling.py \
  --cpt_ckpt ../out/msa_cpt_paper.pth \
  --queries 200 --n_list "8,16,32,64,128,256,512,1024" \
  --top_k 16 --out_json ../out/scaling_curve.json
```

---

## Reported results

**4-hour CPT, single H200, MS MARCO v2.1, 4.98 B tokens (3.14 % of paper's 158.95 B):**

| | pre-CPT | post-CPT |
|---|---|---|
| Held-out router top-1 hit-a-positive (N=64) | 1.3 % | **99.8 %** |
| Held-out LM loss | 3.36 | **2.03** |
| Aux loss | 4.17 (≈ chance log 64) | **0.068** |

**Scaling-curve (held-out, N=8 → 1024 docs = 2 K → 256 K context):**
post-CPT LM loss moves 2.05 → 2.04 (−0.5 %); router top-1 100 % → 98.9 % (−1.1 %).
Pre-CPT collapses from 9 % top-1 @ 2 K context to 0 % at 64 K+ (tracks chance `k/N`).

Details in [`SUMMARY_CN.md`](./SUMMARY_CN.md). Known limitations in the same file.

---

## Status vs. paper

**Aligned:** architecture, losses, LR schedule, dataset/target format, Eq. 5 multi-positive.

**Compute-bound gaps:** backbone is MiniMind-64M (paper uses Qwen3-4B, ~60× bigger);
training saw 4.98 B tokens (paper 158.95 B, ~30× gap); context budget per sample is
16 K (paper extrapolates from 64 K to 100 M). See
[`PLAN_CN.md`](./PLAN_CN.md) for the full remaining-work roadmap.

---

## References

- MSA paper — arXiv [2603.23516](https://arxiv.org/abs/2603.23516)
- Official reference implementation (inference-only) — [EverMind-AI/MSA](https://github.com/EverMind-AI/MSA)
- MiniMind — [jingyaogong/minimind](https://github.com/jingyaogong/minimind)

## License

Apache-2.0 for this repository's code. Upstream projects retain their own licenses.
