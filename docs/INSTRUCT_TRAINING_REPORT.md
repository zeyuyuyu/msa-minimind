# Qwen3.5-9B-Instruct · MSA 训练全流程报告

> **范围**: 第二条训练链路 — 从 `Qwen3.5-9B-Instruct` 出发,因为 base 链的最终分数 1.503/5 被诊断有 5 个错误教训(见 [`BASE_TRAINING_REPORT.md`](BASE_TRAINING_REPORT.md) §7),换 backbone + 加大 LoRA + 用 paper-aligned 大语料重新尝试。
>
> **当前状态**: Phase 5a (CPT 51000 step) 完成,Phase 5b (CPT 续训 60000 step) 跑到 30000 step 后被人工 kill(因为 LM eval F1 在 step 16k–28k 之间一直 flat ≈ 0.12,确认无效训练)。Fork SFT-S1+S2 的 1k step smoke 已完成,**还未做 LLM-judge 全 bench 验证**,我们仍在做 hybrid-oracle 验证以归因。
>
> **关键发现**: 这条链路在 base 链失败后开启,目的是**对照实验**而非完整复现。CPT 阶段的 loss flat 现象 ≈ 「100M token 仅相当于 paper 158B token 的 0.16%」+ 「LR 已经被砍半到 1e-5」,LM 缺乏可学的新增信号 → **大规模 CPT 在不到 1B token 的预算下无法发挥作用**,这是本链路最大的教训。

---

## 0. 训练链路总览

```text
              ┌────────────────────────────────────────┐
              │  Qwen3.5-9B-Instruct (HF 27 GB)        │
              └────────────────────────────────────────┘
                            │
                            │  Phase 5a · CPT shard data (KaLM + ST mix)
                            │  launch_p5_cpt.sh, warmup 2000 + main 120000
                            │  实际跑到 step 51000 后被 stop
                            │  ~437 M tokens (0.275% of 158.95 B target)
                            │  ~94 hr H200
                            ▼
              ┌────────────────────────────────────────┐
              │  p5_shard1_step51000.pt                │
              │  (router + LoRA r=64 接近 paper budget) │
              └────────────────────────────────────────┘
                            │
                            │  Phase 5b · CPT 续训 (paper-aligned shard)
                            │  launch_p5b_cpt.sh, warmup 1000 + main 60000
                            │  实际跑到 step 30000 后被 kill
                            │  ~259 M tokens (0.163% of 158.95 B)
                            │  ~58 hr H200, lm loss FLAT 0.34-0.43
                            ▼
              ┌────────────────────────────────────────┐
              │  p5b_shard1_step30000.pt               │
              │  (LM eval F1 flat 0.12 持续 16k step)  │
              └────────────────────────────────────────┘
                            │
                            │  Fork SFT-S1 (smoke)  → william-dev H200
                            │  launch_fork_s1.sh, 1000 step
                            ▼
              ┌────────────────────────────────────────┐
              │  fork_s1_step1000.pt                   │
              │  (mini-验证: 100% empty, 0% reached_C) │
              └────────────────────────────────────────┘
                            │
                            │  Fork SFT-S2 (curriculum 64→256)  → 已 kill
                            │  launch_fork_s2.sh, 1000 step planned
                            │  (确认 SFT-S1 没有学到三段格式 → SFT-S2 不会改善)
                            ▼
                       (训练链 STOP, 转到 hybrid-oracle 验证)
```

| 阶段 | wall time | tokens | 状态 |
|---|---|---|---|
| Phase 5a CPT | ~94 hr | 437 M / 158.95 B target = 0.275% | step 51000 ckpt 保留 |
| Phase 5b CPT | ~58 hr | 259 M / 158.95 B = 0.163% | step 30000 ckpt 保留,**人工 kill**(lm flat) |
| Fork SFT-S1 | ~3 hr | ~5 M | step 1000 ckpt 保留(smoke) |
| Fork SFT-S2 | 启动 → 25 step → kill | ~0.1 M | 跑了 25 step 因要让出 GPU 做 hybrid 验证 |

---

## 1. Phase 5a · 大规模 CPT(paper-aligned 通用 shard)

### 1.1 数据集

| 项 | 值 |
|---|---|
| 数据来源 | `/workspace/cpt_shards/shard_1/`(自建,见下方 query mix) |
| Query mix(paper §3.4 复现) | KaLM + 33 个 sentence-transformers(SBERT, m3e, BGE, Stella, NV-Embed-v2, Linq, GTE, …),每个数据集 cap 500K → 总 18.06M queries |
| Corpus 文件 | `corpus.txt` 22 344 573 行(2.69 B tokens 估计) |
| Sample 文件 | `samples.jsonl` 25 900 209 行 |
| 单条 sample 形态 | 1 query + 32 candidates(positives + 随机 negs) |
| 文档 token | `--max_doc_len 256` |
| Query token | `--max_query_len 384` |
| Sample 总 tokens | 32 × 256 + 384 = **8576 tokens** |
| Paper budget(参考) | 158.95 B 总 tokens(用于 158.95 B / 8576 ≈ 18.5 M sample 训完 1 epoch) |

### 1.2 训练范式 — Two-Phase + 大幅 scaled-up

跟 base 链的 long_cpt 同 schedule(warmup + main),但所有规模指标都 ×10:

- **Warmup 2000 steps**(base = 1000): `lm_coef=0.1, aux_coef=1.0`,LR `lora=1e-4, router=2e-4`
- **Main 120000 steps**(base = 6000): `lm_coef=1.0, aux_coef=0.1`,LR `lora=2e-5, router=5e-5`

### 1.3 LoRA 配置 — 大幅放大

```text
LoRA injected: r=64 alpha=128.0 dropout=0.0 wrapped=248 linears
Trainable: 273.78 M / total 9227.58 M (2.967%)
  router : 100.66 M
  lora   : 173.11 M
```

> base 链是 r=16/α=32 (43.28M LoRA),instruct 链 r=64/α=128 (173.11M LoRA),**LoRA 容量 ~4×**。这是为了让 LoRA 有足够 rank 同时学「instruct 自带答题倾向不被覆盖」+「MSA 三段输出格式」。

### 1.4 完整 launch 命令

完整脚本: [`scripts/launch_p5_cpt.sh`](../scripts/launch_p5_cpt.sh)

```bash
python -m msa.train_msa_cpt_qwen3_5 \
  --data shard \
  --shard_dir /workspace/cpt_shards/shard_${SHARD} \
  --qwen3_5_path /workspace/qwen35_instruct \
  --num_docs 32 --max_doc_len 256 --max_query_len 384 \
  --msa_chunk_size 64 --msa_top_k 16 --msa_max_docs 64 \
  --num_router_heads 8 --router_k_uses_kv_heads 1 \
  --lora_r 64 --lora_alpha 128 \
  --batch_size 1 --num_workers 2 --epochs 1 \
  --warmup_steps 2000 --main_steps 120000 \
  --warmup_lora_lr 1e-4 --warmup_router_lr 2e-4 \
  --main_lora_lr 2e-5 --main_router_lr 5e-5 \
  --grad_clip 1.0 --log_interval 25 --ckpt_every 1000 \
  --dtype bf16 --assert_sparse \
  --save_dir $RUN_DIR --save_name p5_shard${SHARD}
```

**关键参数差异(vs base CPT)**:

| 参数 | base CPT | Phase 5a CPT | 倍数 |
|---|---|---|---|
| warmup_steps | 1000 | 2000 | 2× |
| main_steps | 6000 | 120000 | 20× |
| warmup_lora_lr | 1e-4 | 1e-4 | 1× |
| main_lora_lr | 1e-5 | **2e-5** | 2× |
| msa_chunk_size | 32 | 64 | 2× |
| msa_top_k | 8 | 16 | 2× |
| max_doc_len | 160 | 256 | 1.6× |
| max_query_len | 192 | 384 | 2× |
| sample tokens | 5312 | **8576** | 1.6× |

### 1.5 Trajectory(Phase 5a)

| Step | Phase | loss | lm | aux | tokens | wall (hr) |
|---|---|---|---|---|---|---|
| 50 | warmup | 3.73 | 1.87 | 3.54 | 0.43 M | 0.09 |
| 200 | warmup | 2.87 | 1.62 | 2.70 | 1.72 M | 0.36 |
| 350 | warmup | 2.37 | 1.68 | 2.20 | 3.00 M | 0.63 |
| 2000 | warmup→main | — | — | — | ~17.2 M | ~3 |
| 10000 | main | ~1.0 | 0.95 | ~0.05 | ~85 M | ~17 |
| 30000 | main | 0.50 | 0.49 | 0.04 | ~257 M | ~50 |
| **51000** | main 中点 stop | **0.37** | **0.37** | **0.02** | **437 M** | **94** |

**输出 ckpt**: `/workspace/runs/p5_shard1/p5_shard1_step51000.pt`(894 MB)

→ Phase 5a stop 决策原因: 51000 step 后 lm loss 趋于 plateau,且为了**让 GPU 给 SFT 用**(后续 fork chain),所以提前 stop。51000 / 120000 = 42.5% 进度。

---

## 2. Phase 5b · CPT 续训(paper-aligned 调小 shard)

### 2.1 数据集变化

Phase 5b 用更小、paper-aligned 的 shard:

| 项 | Phase 5a | Phase 5b |
|---|---|---|
| Shard dir | `cpt_shards/shard_1` | `cpt_shards_paper/shard_1` |
| Corpus size | 22 344 573 lines | **6 787 030 lines (-70%)** |
| Sample count | 25 900 209 | **6 020 157 (-77%)** |
| Token estimate | 2.69 B | **1.48 B (-45%)** |
| Query cap | 无 | `query_cap: 500000`(paper §3.4 fairness) |

### 2.2 训练配置变化

完整脚本: [`scripts/launch_p5b_cpt.sh`](../scripts/launch_p5b_cpt.sh)

```bash
python -u -m msa.train_msa_cpt_qwen3_5 \
  --data shard \
  --shard_dir /workspace/cpt_shards_paper/shard_${SHARD} \
  --qwen3_5_path /workspace/qwen35_instruct \
  --resume_from $RESUME              # ← p5_shard1_step51000.pt
  --num_docs 32 --max_doc_len 256 --max_query_len 384 \
  --msa_chunk_size 64 --msa_top_k 16 --msa_max_docs 64 \
  --num_router_heads 8 --router_k_uses_kv_heads 1 \
  --lora_r 64 --lora_alpha 128 \
  --batch_size 1 --num_workers 2 --epochs 1 \
  --warmup_steps 1000 --main_steps 60000 \
  --warmup_lora_lr 5e-5 --warmup_router_lr 1e-4 \
  --main_lora_lr 1e-5 --main_router_lr 2.5e-5 \   # ← 砍半
  --grad_clip 1.0 --log_interval 50 --ckpt_every 2000 \
  --dtype bf16 --gradient_checkpoint 1
```

**关键变化**:

- `--resume_from p5_shard1_step51000.pt`(从 5a 续承)
- `--main_steps 60000`(原 120000 的一半)
- `--main_lora_lr 1e-5` / `--main_router_lr 2.5e-5`(LR 砍半,因为接的是已 warmed ckpt)
- `--ckpt_every 2000`(原 1000 的两倍,节省磁盘)

### 2.3 Trajectory(Phase 5b)

| Step | Phase | loss | lm | aux | tokens(累计) | wall(hr) |
|---|---|---|---|---|---|---|
| 50 | warmup | 0.36 | 0.84 | 0.28 | 0.43 M | 0.09 |
| 200 | warmup | 0.20 | 0.63 | 0.14 | 1.72 M | 0.37 |
| 1000 | warmup → main 切换 | — | — | — | ~8.6 M | 1.5 |
| 4000 | main | ~0.4 | 0.39 | 0.10 | ~35 M | ~7 |
| 16000 | main | 0.42 | 0.40 | 0.13 | ~140 M | ~31 |
| 20000 | main | 0.40 | 0.39 | 0.12 | ~177 M | ~37 |
| 26000 | main | 0.42 | 0.40 | 0.13 | ~225 M | ~46 |
| 28000 | main | 0.41 | 0.40 | 0.13 | ~241 M | ~48 |
| **30000** | **main, 人工 kill** | **0.39** | **0.38** | **0.10** | **259 M (0.163%)** | **57.9** |

→ **lm loss 在 step 16000 → 30000 之间长期 flat 在 0.34–0.43 之间震荡**。Mini-验证(`scripts/bench_evermind_aligned_v2.py` × 1 bench × 20 query)在 step 16k/20k/24k/28k 上 LM F1 都 ≈ 0.12,**没有进步**。

→ **Kill 原因**(May 7 04:30 UTC 由 user 决定):
1. 验证 F1 flat 16000+ step 无改善,继续训练浪费算力
2. 释放 H100 给 hybrid-oracle 验证(诊断 base 链 1.503 分根因)
3. 如果要重训,应该先解决「为什么 loss flat」而不是堆 step

**输出 ckpt 保留**:
- `/workspace/runs/p5b_shard1/p5b_shard1_step30000.pt` (894 MB)
- `step28000.pt`、`step26000.pt`(回滚备用)

---

## 3. Fork SFT-S1(william-dev H200, smoke 1k step)

### 3.1 设计动机

Phase 5b CPT 跑 30k step lm flat 不知道能不能恢复;不想等 60k step 跑完才发现没救。设计一个**廉价 fork**:

- 在另一台机器(william-dev H200)上,从 Phase 5b step 20000 ckpt 续承(挑相对早的、还能动的)
- 跑 1000 step SFT-S1(smoke 量级,3 hr 完成)
- 如果 SFT 后 mini-验证 LLM-judge ≥ 2.5 → CPT route 还有救,继续 60000 step 跑完
- 如果 SFT 后 LLM-judge 远低于 2.5 → 整条 instruct chain 不值得继续

### 3.2 数据集

完全沿用 base 链 SFT 的 `sft_mix`,但每个子集 `--sft_max_per` 从 20000 调到 5000(smoke):

```bash
--data sft_mix \
--sft_datasets "hotpotqa,musique,triviaqa,nq,msmarco" \
--sft_max_per 5000
```

### 3.3 训练配置

完整脚本: [`launch_fork_s1.sh`](../scripts/launch_fork_s1.sh)(本地 fork 在 `/workspace/launch_fork_s1.sh` on william-dev)

```bash
python -u -m msa.train_msa_sft_qwen3_5 \
  --qwen3_5_path /workspace/qwen35_instruct \
  --resume_ckpt /workspace/ckpt/p5b_step20k.pt \      # ← 从 5b step20k 续
  --data sft_mix \
  --sft_datasets "hotpotqa,musique,triviaqa,nq,msmarco" \
  --sft_max_per 5000 \
  --num_docs 64 --max_doc_len 96 --max_query_len 1024 \
  --msa_chunk_size 32 --msa_top_k 8 --msa_max_docs 64 \
  --num_router_heads 8 --router_k_uses_kv_heads 1 \
  --lora_r 64 --lora_alpha 128 \                     # ← 与 5b 一致
  --batch_size 1 --epochs 5 --total_steps 1000 \
  --lr_warmup_steps 50 \
  --lora_lr 5e-5 --router_lr 1e-4 \
  --lm_coef 1.0 --aux_coef_start 0.5 --aux_coef_end 0.1 \
  --grad_clip 1.0 --ckpt_every 100 \                 # ← 高频 ckpt 给 daemon 验证
  --dtype bf16 --gradient_checkpointing --assert_sparse
```

**与 base 链 SFT-S1 的区别**:

| 配置项 | base SFT-S1 | Fork SFT-S1 |
|---|---|---|
| backbone | qwen35_base | **qwen35_instruct** |
| resume_ckpt | qwen3_5_msa_long_step7000.pt | **p5b_step20k.pt** (instruct CPT) |
| LoRA | r=16/α=32 | **r=64/α=128** |
| sft_max_per | 20000 | **5000** (smoke) |
| total_steps | 7000 | **1000** (smoke) |
| ckpt_every | 500 | **100** (高频,配合 auto_eval daemon) |

### 3.4 Trajectory(Fork SFT-S1)

LM loss 走势:

| step | lm loss |
|---|---|
| 25 | 1.11 |
| 100 | 0.85 |
| 300 | 0.62 |
| 500 | 0.55 |
| 700 | 0.50 |
| 1000 | **0.48** |

→ **LM loss 1.11 → 0.48,降了 57%** — 模型确实在学习。

### 3.5 Mini-验证(fork SFT-S1 step 1000)

跑 3 bench × 20 query 的轻量验证 + LLM-judge:

```text
nq_score: nan       (生成全 empty,LLM-judge 跳过)
hp_score: nan
tv_score: nan
router_prec: nan
empty_rate: 1.00    ← 100% empty answer
reached_part_c_rate: 0.00   ← 0% 模型触及 <End-of-Retrieve>
```

**结论**: Fork SFT-S1 step 1000 的输出 100% 不符合 MSA 三段格式。

### 3.6 当时的诊断错误(自我反省)

我当时(May 6)看到 `reached_part_c_rate=0.00` 直接判定「SFT-S1 不教三段格式」,这是**错的**。事后查代码:

- `msa/dataset_msa.py::_make_prompt_and_target` 是**所有 SFT 阶段共用的 target builder**,SFT-S1 和 SFT-S2 都教 part_a/b/c。
- 真正原因: 1000 step + LoRA r=64 不足以**把三段格式从无到有学到 instruct backbone 上**。
  - 旧 base 链 SFT-S1 跑了 7000 step + r=16 才把三段学好(在 base backbone 上)
  - instruct backbone 已经被 SFT/RLHF 训过,「自然回答」习惯根深蒂固,要 override 它输出 part_a/b/c 需要更多步数

→ 当时的错误判断让我立刻又启动 Fork SFT-S2(认为 S1 不教格式,S2 教),其实**应该把 fork SFT-S1 跑到至少 5000 step 再判断**。

---

## 4. Fork SFT-S2(启动 25 step 后 kill)

### 4.1 设计意图

基于 §3.6 的错误诊断,启动 S2 想用 curriculum + S2 特有的 「num_docs 增长」目标教三段格式。脚本: [`launch_fork_s2.sh`](../scripts/launch_fork_s2.sh)

```bash
--resume_ckpt /workspace/runs/p5d_fork_s1_instruct/qwen3_5_msa_sft_s1_step1000.pt \
--total_steps 1000 \
--msa_max_docs 256 \                        # ← 比 fork S1 大
--aux_coef_start 0.3 --aux_coef_end 0.05    # ← 比 fork S1 弱
```

### 4.2 实际运行: 25 step → kill

跑到 step 25 时 lm = 0.5365、GPU util 97%。被 kill 原因:

- 用户 explicit 决定: 「把 H100 让给 root cause 实验和后续从头重训」
- Fork SFT-S2 训完(预计 ~25 min)也只能再跑一次 mini-验证,信息量 < 直接做 hybrid-oracle 验证
- **当前优先级**: 先归因 base 链 1.503 分的根因,再决定 instruct chain 怎么重训

---

## 5. 与 Base 链对比

| 维度 | Base 链 | Instruct 链(本报告) |
|---|---|---|
| Backbone | Qwen3.5-9B-Base | Qwen3.5-9B-Instruct |
| CPT 数据 | MS MARCO v2.1 5万 facts | KaLM + 33 ST 18M queries(paper-aligned) |
| CPT tokens | ~5 B(7000 step × 5312 tok) | 实际 437 M (5a) + 259 M (5b) = 696 M |
| LoRA | r=16/α=32(43 M trainable) | r=64/α=128(173 M trainable, **4×**) |
| MSA chunk size | 32 | 64 |
| MSA top_k | 8 | 16 |
| 训练总时长 | ~66 hr(端到端跑完 SFT-S2) | ~155 hr(只跑 CPT 一半) |
| 完整产物 | SFT-S2 final ckpt | 仅 CPT step30k + fork SFT-S1 step1k(smoke) |
| 最终 LLM-judge | **1.503 / 5.0**(已知) | **未跑全 bench 验证** |

---

## 6. 这条链路目前学到的教训

### 教训 ① · 大 CPT 在小预算下没用

paper 用 158.95 B token CPT,我们 instruct 链 5a + 5b 加起来才 0.696 B token = **0.44%**。lm loss 在 0.34–0.42 之间 flat,因为已经被 instruct SFT/RLHF 训过的 LM **没有可学的新增信号**:

- MSA 层的 router 在 CPT-warmup 1000 step 内 aux loss 已经从 0.28 降到 0.05,**router 早就饱和**
- LM 层的 LoRA 对 instruct backbone 修改有限,**lm loss 看的是 query token 上的 next-token CE**,而 next-token 大多是 doc id `[gid]` 或 doc text(part_b),这些在 instruct 模型里本身就 well-defined → 没什么可学

→ **结论**: 在 instruct backbone 上做大规模 CPT,**除非 token 预算 ≥ paper 量级(150B+),否则 lm loss 只会 flat**。要么 (a) 就别 CPT,直接 SFT;要么 (b) 拉到 paper 量级。

### 教训 ② · Fork SFT smoke 步数下限要 ≥ 5000 step

base 链 SFT-S1 用 r=16/α=32 跑 7000 step 才把三段格式学进去(LM loss 1.25 → 0.44)。fork 链 用 r=64/α=128 跑 1000 step(lm 1.11 → 0.48)虽然 loss 下降很快,但生成时 100% 不符合三段格式 — 这意味着 **loss 下降仅反映 LoRA 在「拷贝 doc text」上 fit 得很快,根本没碰「输出三段格式」的 token sequence**。

→ **结论**: 任何 SFT smoke 都必须 step ≥ 5000(同 base 链 S1 step5000 时 lm ≈ 0.5),否则验证结果是 noise。

### 教训 ③ · 边训边验证必须有红线门槛

之前 base 链跑完 6000 step SFT-S2 才知道结果 1.50。本次 instruct 链已经引入 [`scripts/auto_eval_daemon.sh`](../scripts/auto_eval_daemon.sh)(william-dev),设了 6 条红线:

| 红线 | 阈值 | 触发 → 行动 |
|---|---|---|
| router precision < 0.05 | 持续 3 ckpt | warning |
| empty_answer_rate > 0.30 | 持续 3 ckpt | stop training |
| reached_part_c_rate < 0.50 | 持续 3 ckpt | warning |
| LLM-judge avg < 1.98 | 持续 3 ckpt | stop |
| 5 ckpt 连续无 improvement | — | stop |
| ckpt 落盘 30 min 无 verify | — | alert |

→ 这是从 base 链 → instruct 链最大的工程改进:**任何后续训练都必须先有验证 daemon 在跑**。

---

## 7. 当前正在做的事(May 7)

### 7.1 Hybrid-Oracle 验证(归因实验)

[`scripts/bench_msa_oracle_hybrid.py`](../scripts/bench_msa_oracle_hybrid.py) 把 base 链的 SFT-S2 final ckpt + **强制 pooled_cache 只装 gold docs**,看 LLM-judge:

- 若 ≈ 3.5 → base 链的 LM 没坏,问题全在 router → **重训 router**(可能换 batch contrastive 或者用更大对比池)
- 若 ≈ 1.5 → LoRA r=16 真把 LM 训坏了 → **重训整个 SFT,r ≥ 64 + 更长 step**

### 7.2 Vanilla Instruct + BM25 验证(公平 baseline)

[`scripts/bench_vanilla_instruct.py`](../scripts/bench_vanilla_instruct.py) 在 william-dev 上跑 vanilla 9B-instruct + BM25 top-5(走真实 retrieval,不是 oracle),看 LLM-judge:

- 若 ≈ 3.5 → 普通 RAG 已经足够好,**MSA 在 < 64K 的 bench 上没有 advantage**(MSA 的真正价值在 100M+ context)
- 若 ≈ 2.0 → 普通 RAG 也不行,MSA 的 router 还有空间证明价值

### 7.3 重训方案规划(待 hybrid 结果)

依据 7.1 / 7.2 的结果,下一轮训练会(初稿):

- **Backbone**: 维持 Qwen3.5-9B-Instruct(但 LoRA r 调整)
- **CPT**: 跳过(教训 ①),直接 SFT
- **SFT 数据**: sft_mix + 加入更多 NQ-style query(教训 base 链 NQ 91% empty)
- **SFT 步数**: ≥ 10000 step(教训 ②)
- **验证 daemon**: 全程跑(教训 ③)
- **目标**: 端到端 LLM-judge ≥ 3.5(超过 vanilla instruct + oracle 减一档,且优于 base 链 1.50)

---

## 8. 完整文件清单

| 类别 | 主机 | 路径 |
|---|---|---|
| Phase 5a CPT 51 ckpt | cvm-rl `msa-dev` | `/workspace/runs/p5_shard1/p5_shard1_step51000.pt` |
| Phase 5b CPT 30k ckpt | cvm-rl `msa-dev` | `/workspace/runs/p5b_shard1/p5b_shard1_step30000.pt` |
| Phase 5b 中间 ckpt | cvm-rl `msa-dev` | `step{4000,6000,...,30000}.pt`,every 2000 |
| Fork SFT-S1 1k ckpt | william-dev `compassionate_austin` | `/workspace/runs/p5d_fork_s1_instruct/qwen3_5_msa_sft_s1_step1000.pt` |
| Fork SFT-S1 中间 ckpt | william-dev | `step100.pt` ~ `step1000.pt`,every 100 |
| Fork SFT-S2 partial ckpt | (kill 后无落盘) | — |
| Mini-验证 trajectory | william-dev | `/workspace/runs/p5d_fork_s1_instruct/eval_trajectory.tsv` |
| 训练日志 | cvm-rl | `/workspace/runs/p5_shard1.log`、`/workspace/runs/p5b_shard1.log` |

---

## 9. 这条链路与 paper 的对照表

| 维度 | MSA Paper §3 | Instruct 链(本报告) | 完成度 |
|---|---|---|---|
| Backbone | Qwen2-7B / Qwen3-8B | Qwen3.5-9B-Instruct | ✓ 略大 |
| CPT corpus | KaLM + ST mix, 158.95 B tokens | 同源,696 M tokens | **0.44%** |
| CPT epoch | 1 | 0.0044 | **0.44%** |
| LoRA r | (unspecified, 推测 ≥ 32) | 64 | ≥ |
| MSA layers | every 4 layer (8 layers in 32) | 同 [3,7,11,15,19,23,27,31] | ✓ |
| router_heads | 8 | 8 | ✓ |
| chunk_size | 32 | 64 | ↑ |
| top_k | 8 | 16 | ↑ |
| SFT-S1 步数 | (unspecified) | 1000 (smoke), 未完整跑 | × |
| SFT-S2 curriculum | 8K→64K | 未跑 | × |
| Final eval bench | 11 benches | 待 hybrid-oracle | — |
| Final LLM-judge | 报 paper Table 2,3 | **未达成** | × |

---

*Last updated: 2026-05-07 — 训练已暂停,等 hybrid-oracle 验证结果决定重训方案。*
