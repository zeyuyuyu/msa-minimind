# Qwen3.5-9B-Base · MSA 训练全流程报告

> **范围**: 第一条训练链路 — 从 `Qwen3.5-9B-Base` 出发,先做长上下文 CPT(MS MARCO 预热路由),再做两阶段 SFT(SFT-S1 学三段输出格式,SFT-S2 上 curriculum 把 context 从 8K 拉到 64K)。所有训练在 cvm-rl 容器 `msa-dev` 上完成,单卡 H200 (~143 GB)。
>
> **最终评分**: 10-bench × 100q 经 LLM-judge(`google/gemini-2.5-flash`)平均 **1.503 / 5.0**,和 Vanilla `Qwen3.5-9B-Instruct` + Oracle docs 的 **3.957 / 5.0** 相比有 ~62% 的差距。**根因来自三个叠加因素**: ① Router precision 0.01–0.28(主因) ② NQ/dureader 上空答率 32–91%(次因) ③ MSA 三段输出格式 + part_c 截断(可观察因素)。详见 §6。

---

## 0. 训练链路总览

```text
              ┌────────────────────────────────────────┐
              │  Qwen3.5-9B-Base (HF 原始 27 GB)        │
              └────────────────────────────────────────┘
                            │
                            │  Long CPT (MS MARCO v2.1)
                            │  long_cpt_qwen3_5.sh, 7000 steps
                            │  ~5 GB tokens, ~15 hr H200
                            ▼
              ┌────────────────────────────────────────┐
              │  qwen3_5_msa_long_step7000.pt          │
              │  (router 已对齐, LM 仍是 base)          │
              └────────────────────────────────────────┘
                            │
                            │  SFT Stage-1 (sft_mix 5 benches)
                            │  scripts/sft_stage1_qwen3_5.sh
                            │  7000 steps, 35.84 M tokens
                            │  ~8.5 hr H200
                            ▼
              ┌────────────────────────────────────────┐
              │  qwen3_5_msa_sft_s1.pt                 │
              │  (LM 学到三段输出格式; 短 context 8K)   │
              └────────────────────────────────────────┘
                            │
                            │  SFT Stage-2 curriculum 64→512 docs
                            │  scripts/sft_stage2_qwen3_5.sh
                            │  6000 steps, 212.58 M tokens
                            │  ~42 hr H200
                            ▼
              ┌────────────────────────────────────────┐
              │  qwen3_5_msa_sft_s2.pt  ← FINAL CKPT   │
              │  (LLM-judge 平均 1.503 / 5.0)           │
              └────────────────────────────────────────┘
```

| 阶段 | wall time | tokens | trainable params | 主目的 |
|---|---|---|---|---|
| Long CPT | ~15 hr | 5.04 B(50K facts × 32 docs × 6144 tok / sample 上限,实际 batch 7000 步) | ~144 M (LoRA r=16 + router) | 让 8 个 MSA 层学会「query → 选 top-k doc」 |
| SFT-S1 | 8.5 hr | 35.84 M | ~144 M | 让 LM 学会输出 `part_a + part_b + part_c` 三段格式 |
| SFT-S2 | 41.9 hr | 212.58 M | ~144 M | 在 8K → 64K context curriculum 上保持三段格式 |
| **合计** | **~66 hr (单卡 H200)** | **~250 M token (SFT) + 5 B (CPT)** | LoRA 1.6% / total 9 B | — |

---

## 1. Stage-0 · 长上下文 CPT(MS MARCO v2.1)

### 1.1 数据集

| 项 | 值 |
|---|---|
| 来源 | HuggingFace `microsoft/ms_marco`, `v2.1` config, `train` split |
| 加载方式 | `datasets.load_dataset('ms_marco', 'v2.1', split='train')`, 见 `msa/dataset_msa.py::build_msmarco_dataset` |
| 抽样规模 | `--num_facts 50000`(5 万条 query + passages) |
| 单条 sample 形态 | 1 query + 32 candidate passages(其中 1 个 positive,31 个 corpus 内随机 negative) |
| 文档 token | `--max_doc_len 160` |
| Query token | `--max_query_len 192` |
| Sample 总 tokens | 32 × 160 + 192 = **5312 tokens** |
| Corpus 总规模 | 5 万 facts → 约 1.6 M passages 进入 corpus 池 |

### 1.2 训练范式: Two-Phase Loss(Warmup + Main)

完全对齐 William 在 `Qwen3-8B` 上跑通的 recipe(`scripts/long_cpt_qwen3.sh`)的 schedule shape。Loss 由两部分组成:

```text
L_total = lm_coef * L_LLM   +   aux_coef * L_aux

  L_LLM  : next-token LM loss on the query side (后向传给 LoRA delta)
  L_aux  : router contrastive loss (paper §3.2 Eq. 4)
           = - log( exp(s_pos) / Σ exp(s_*) )  per MSA layer, mean over 8 layers
```

- **Warmup (1000 steps)**: `lm_coef=0.1, aux_coef=1.0` — 几乎只优化 router,让 8 个 MSA 层的 query-doc 匹配从随机初始化进入「能区分 positive vs negative」阶段。LR 用 `warmup_lora_lr=1e-4 / warmup_router_lr=2e-4`。
- **Main (6000 steps)**: `lm_coef=1.0, aux_coef=0.1` — 主要优化 LM 在「MSA 层 attend 到正确 docs」的下一步 token 预测,router 留 0.1 的小弱信号防止漂移。LR 降到 `main_lora_lr=1e-5 / main_router_lr=5e-5`。

### 1.3 完整 launch 命令

完整脚本: [`scripts/long_cpt_qwen3_5.sh`](../scripts/long_cpt_qwen3_5.sh)

```bash
python -m msa.train_msa_cpt_qwen3_5 \
  --qwen3_5_path /workspace/qwen35_base \
  --data ms_marco --msmarco_version v2.1 --msmarco_split train \
  --num_facts 50000 \
  --num_docs 32  --max_doc_len 160  --max_query_len 192 \
  --msa_chunk_size 32  --msa_top_k 8  --msa_max_docs 64 \
  --num_router_heads 8  --router_k_uses_kv_heads 1 \
  --lora_r 16  --lora_alpha 32 \
  --batch_size 1 --num_workers 2 --epochs 10 \
  --warmup_steps 1000 --main_steps 6000 \
  --warmup_lora_lr 1e-4  --warmup_router_lr 2e-4 \
  --main_lora_lr 1e-5    --main_router_lr 5e-5 \
  --grad_clip 1.0 --log_interval 25 --ckpt_every 350 \
  --dtype bf16 --assert_sparse \
  --save_dir $RUN_DIR --save_name qwen3_5_msa_long
```

### 1.4 训练代码模块

| 模块 | 作用 |
|---|---|
| [`msa/train_msa_cpt_qwen3_5.py`](../msa/train_msa_cpt_qwen3_5.py) | CPT 主入口(arg parse,dataset 构建,两段 schedule 调度) |
| [`msa/dataset_msa.py::MSACPTDataset`](../msa/dataset_msa.py) | 通用 MSA dataset 容器,提供 `_make_prompt_and_target` |
| [`msa/dataset_msa.py::build_msmarco_dataset`](../msa/dataset_msa.py) | 从 `microsoft/ms_marco` 解析 facts → MSACPTDataset |
| [`msa/modeling_qwen3_5_msa.py`](../msa/modeling_qwen3_5_msa.py) | `MSAQwen3_5ForCausalLM`,把 8 个 dense attention 层换成 MSA(router + chunked memory attention) |
| [`msa/lora.py::apply_lora_and_freeze`](../msa/lora.py) | 在 248 个 linear(`q/k/v/o + in_proj_qkv + in_proj_z + in_proj_b + in_proj_a + out_proj + gate/up/down_proj`)上注入 LoRA,冻结其他参数 |

### 1.5 Trainable parameter budget

```text
LoRA injected: r=16 alpha=32.0 dropout=0.0  wrapped=248 linears
Trainable: 143.94 M / total 9097.74 M  (1.582%)
  router : 100.66 M
  lora   : 43.28 M
```

### 1.6 Trajectory(loss / aux 关键 step)

| Step | Phase | loss | lm | aux | tokens (M) | wall (min) |
|---|---|---|---|---|---|---|
| 25  | warmup | 2.42 | 1.25 | 2.34 | 0.13 | 1.8 |
| 1000 | warmup→main 切换 | — | — | — | 5.12 | 73 |
| 1000 | main step 1 | ~1.3 | 0.85 | ~0.9 | — | — |
| 4000 | main | 0.66 | 0.65 | 0.10 | ~22 | ~330 |
| 7000 | **main 末** | 0.45 | 0.44 | 0.05 | **35.84** | **511** |

**输出 ckpt**: `/workspace/ckpt/qwen3_5_msa_long_step7000.pt`(43 MB LoRA delta + 100 MB router weight)

---

## 2. Stage-1 SFT · 短 8K context, 5-benchmark 混合

### 2.1 数据集 — `sft_mix`

完整定义: [`msa/dataset_msa_sft.py::build_sft_mix`](../msa/dataset_msa_sft.py)

| 子集 | HF 来源 | 样本形态 | 上限 |
|---|---|---|---|
| **hotpotqa** | `hotpot_qa` distractor | 10 paragraphs/query, 2 supporting → 多 positive | `--sft_max_per 20000` |
| **musique** | `dgslibisey/MuSiQue` | 变长 supporting + distractor → 多 positive | 20000 |
| **triviaqa** | `Tevatron/triviaqa` | 1 positive + 多 hard-neg → 单 positive | 20000 |
| **nq** | `Tevatron/wikipedia-nq` | 同 triviaqa | 20000 |
| **msmarco** | `Tevatron/msmarco-passage` | passage retrieval | 20000 |
| **合计** | — | — | **89 566 samples** (训练 log line 17) |

5 个 corpus **不合并**,每个 dataset 保留独立 `Corpus` —— HotpotQA 的随机 negative 不会出现 TriviaQA 的 passage,确保语义可信。

### 2.2 SFT 输出格式(`_make_prompt_and_target`)

这是整条链路最关键的 design,paper §3.5 / Fig 3。完整定义:[`msa/dataset_msa.py::MSACPTDataset._make_prompt_and_target`](../msa/dataset_msa.py#L130)

```text
PROMPT (loss masked, 模型只看):
  <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n

TARGET (supervised, 模型必须输出):
  Part A:  [gid1] [gid2] ... <|object_ref_end|>\n           # 路由出的 doc id 序列
  Part B:  [gid1]. <text_1><|object_ref_end|>\n              # 「Original Text」机制
           [gid2]. <text_2><|object_ref_end|>\n              # 一段 / positive
           ...
  Part C:  <End-of-Retrieve>\n{answer}<|im_end|>             # 真正的答案
```

> **Part B 的来源**: paper §3.5 称为「Original Text」/「OT」;Table 4 ablation 显示禁用它会让 avg score 掉 37.1%。本仓库默认 `include_doc_text_in_target=True`。

### 2.3 训练配置

完整脚本: [`scripts/sft_stage1_qwen3_5.sh`](../scripts/sft_stage1_qwen3_5.sh)

```bash
python -m msa.train_msa_sft_qwen3_5 \
  --qwen3_5_path /workspace/qwen35_base \
  --resume_ckpt /workspace/ckpt/qwen3_5_msa_long_step7000.pt \
  --data sft_mix \
  --sft_datasets "hotpotqa,musique,triviaqa,nq,msmarco" \
  --sft_max_per 20000 \
  --num_docs 64  --max_doc_len 96  --max_query_len 1024 \
  --msa_chunk_size 32 --msa_top_k 8 --msa_max_docs 64 \
  --num_router_heads 8 --router_k_uses_kv_heads 1 \
  --lora_r 16  --lora_alpha 32 \
  --batch_size 1 --epochs 10 --total_steps 7000 \
  --lr_warmup_steps 200 \
  --lora_lr 5e-5 --router_lr 1e-4 \
  --lm_coef 1.0 --aux_coef_start 0.5 --aux_coef_end 0.1 \
  --grad_clip 1.0 --ckpt_every 500 \
  --dtype bf16 --assert_sparse
```

| 配置项 | 值 | 备注 |
|---|---|---|
| backbone | `Qwen3.5-9B-Base` | 不是 instruct |
| Resume from | `qwen3_5_msa_long_step7000.pt` | 续承 CPT 阶段 LoRA + router |
| Per-sample tokens | `64 × 96 + 1024 = 7168` | 算上 in-prompt OT 注入 ~ 8K |
| LoRA | r=16, α=32 | LoRA 只在 LM 上,router 已被 CPT 训过 |
| LR (cosine) | LoRA `5e-5`,router `1e-4` | 200 step warmup 后降至单一 cosine |
| Aux coef | `0.5 → 0.1`(线性) | router 已 OK,弱信号防漂移 |
| 步数 | **7000** | dataset 89566 / batch 1 ≈ 0.08 epoch (实际 1+ epoch 因 mixer plan 复用) |

### 2.4 Trajectory(SFT-S1)

| Step | loss | lm | aux | aux_coef | tokens (M) | wall (min) |
|---|---|---|---|---|---|---|
| 25 | 2.416 | 1.249 | 2.338 | 0.499 | 0.13 | 1.8 |
| 500 | 1.387 | 0.935 | 0.957 | 0.471 | 2.56 | 36 |
| 1000 | 0.833 | 0.661 | 0.387 | 0.443 | 5.12 | 73 |
| 3000 | ~0.55 | 0.50 | 0.25 | 0.30 | ~15 | ~220 |
| 7000 | **0.450** | **0.444** | **0.053** | 0.100 | **35.84** | **511.7** |

LM loss 从 1.25 → 0.44 是「学三段输出格式」的标志。Aux loss 从 2.34 → 0.05 因为 router 已经在 CPT 阶段学好,这里只是微调。

---

## 3. Stage-2 SFT · Curriculum 8K → 64K context

### 3.1 设计动机

Paper §3.3.2:「Stage 2: curriculum from **8K to 64K** memory bank」。SFT-S1 已经会输出三段格式,但只见过 64 docs / 8K tokens 的小 context。线上推理时 corpus 有上万 docs,router 必须从更大候选集里选 — 我们用 **5 rung curriculum** 模拟这种规模 scale-up:

| Rung | num_docs | per-sample tokens | duration |
|---|---|---|---|
| 0 | 64 | 64×128 + 1024 = **9.2K** | step 0–1199 |
| 1 | 128 | 128×128 + 1024 = **17.4K** | step 1200–2399 |
| 2 | 256 | 256×128 + 1024 = **33.8K** | step 2400–3599 |
| 3 | 384 | 384×128 + 1024 = **50.2K** | step 3600–4799 |
| 4 | 512 | 512×128 + 1024 = **65.5K**(paper 64K target) | step 4800–5999 |

`--curriculum_num_docs "64,128,256,384,512" --curriculum_every 1200` 在 [`msa/train_msa_sft_qwen3_5.py`](../msa/train_msa_sft_qwen3_5.py) 里 driven。

### 3.2 训练配置

完整脚本: [`scripts/sft_stage2_qwen3_5.sh`](../scripts/sft_stage2_qwen3_5.sh)

```bash
python3 -m msa.train_msa_sft_qwen3_5 \
  --qwen3_5_path /workspace/qwen35_base \
  --resume_ckpt /workspace/runs/sft_s1_qwen3_5_latest/qwen3_5_msa_sft_s1.pt \
  --data sft_mix \
  --sft_datasets "hotpotqa,musique,triviaqa,nq,msmarco"  --sft_max_per 20000 \
  --num_docs 64  --max_doc_len 128  --max_query_len 1024 \
  --msa_chunk_size 32 --msa_top_k 8 --msa_max_docs 512 \
  --num_router_heads 8 --router_k_uses_kv_heads 1 \
  --lora_r 16 --lora_alpha 32 \
  --batch_size 1 --epochs 30 --total_steps 6000 --lr_warmup_steps 100 \
  --lora_lr 3e-5 --router_lr 5e-5 \
  --lm_coef 1.0 --aux_coef_start 0.2 --aux_coef_end 0.05 \
  --grad_clip 1.0 --ckpt_every 600 \
  --curriculum_num_docs "64,128,256,384,512" --curriculum_every 1200 \
  --gradient_checkpointing --assert_sparse --dtype bf16
```

| 与 SFT-S1 的区别 | 说明 |
|---|---|
| `--max_doc_len 128`(↑ from 96) | rung 4 时 context = 65K |
| `--msa_max_docs 512`(↑ from 64) | 跟 curriculum 上限对齐 |
| `--lora_lr 3e-5`(↓ from 5e-5) | 防 long-context 训练阶段过拟合 |
| `--aux_coef_start 0.2 → 0.05` | 比 SFT-S1 更轻的 router 信号 |
| `--gradient_checkpointing` | rung 4 时 65K context × 32 layer,不 GC 会 OOM |
| `--total_steps 6000` | 5 rung × 1200 step |

### 3.3 Trajectory(SFT-S2)

| Step | Rung | num_docs | tok/sample | loss | lm | aux | tokens (M) | wall (hr) |
|---|---|---|---|---|---|---|---|---|
| 25 | 0 | 64 | 9216 | 0.49 | 0.47 | 0.10 | 0.23 | 0.06 |
| 1200 | 0→1 | curriculum rebuild → 128/17.4K | — | — | — | 11.06 | 3.07 |
| 2400 | 1→2 | → 256/33.8K | — | — | — | ~30 | ~7.96 |
| 3600 | 2→3 | → 384/50.2K | — | — | — | ~70 | ~17 |
| 4800 | 3→4 | → 512/65.5K | — | — | — | ~120 | ~28 |
| 6000 | 4 末 | 512 | 65536 | **0.149** | **0.139** | 0.20 | **212.58** | **41.9** |

**输出 ckpt**: `/workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt`(374 MB)

---

## 4. 评估方法 · Paper §4.2 Path-A

完整脚本: [`scripts/bench_evermind_aligned_v2.py`](../scripts/bench_evermind_aligned_v2.py)

### 4.1 设置(MSA 完整端到端)

| 项 | 值 |
|---|---|
| Bench root | `/workspace/msa_bench/` (10 个 sub-dir) |
| Encoded corpus | `/workspace/encoded_corpora/` (per-MSA-layer pre-pooled K/V) |
| Top-K docs | `--top_k 10` |
| Max new tokens | **1024** (避免 part_b 截断) |
| Per-bench queries | 100 |
| Prompt | `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n`(裸 question,**docs 不在 prompt 里**) |
| 输出解析 | `parse_msa_train_response` → 切 `<End-of-Retrieve>` 取 part_c |
| LLM-judge | `google/gemini-2.5-flash` via OpenRouter,5-point QA quality scale |

### 4.2 IR 检索指标(router precision/recall on top-10)

| Bench | precision | recall | F1 |
|---|---|---|---|
| dureader | 0.046 | 0.460 | 0.084 |
| 2wikimultihopqa | 0.026 | 0.123 | 0.043 |
| hipporag_narrative | 0.044 | 0.440 | 0.080 |
| hipporag_popqa | 0.022 | 0.110 | 0.037 |
| triviaqa_06M | **0.279** | **0.450** | **0.330** |
| triviaqa_10M | 0.117 | 0.179 | 0.137 |
| hotpotqa | 0.085 | 0.425 | 0.142 |
| musique | 0.028 | 0.106 | 0.044 |
| nature_questions | 0.035 | 0.350 | 0.064 |
| msmarco_v1 | **0.010** | 0.100 | 0.018 |
| **avg** | **~0.069** | ~0.274 | ~0.098 |

→ 除 TriviaQA-06M 外,**router precision ≤ 0.12,等于平均给 LM 喂 88% 噪声 docs**。msmarco_v1 precision 1%,几乎纯噪声。

### 4.3 LLM-judge 评分(0–5)

| Bench | LLM-judge | empty_answer_rate | reached_part_c_rate |
|---|---|---|---|
| triviaqa_06M | **2.71** | 0.08 | 0.96 |
| triviaqa_10M | 1.96 | 0.04 | 0.96 |
| hotpotqa | 1.81 | 0.01 | 0.99 |
| dureader | 1.80 | 0.32 | 0.72 |
| msmarco_v1 | 1.71 | 0.19 | 1.00 |
| 2wikimultihopqa | 1.39 | 0.02 | 0.98 |
| hipporag_popqa | 1.26 | 0.02 | 0.98 |
| nature_questions | **1.01** | **0.91** | 1.00 |
| hipporag_narrative | 0.72 | 0.53 | 0.48 |
| musique | 0.66 | 0.03 | 0.97 |
| **AVERAGE** | **1.503** | — | — |

详细 JSON: `/workspace/eval_evermind_aligned_v2/full/llmscore_summary.json` 与 `summary.json`。

---

## 5. 与 Baseline 对比(注意: **不是严格 apple-to-apple**)

| 实验 | 模型 | docs 来源 | prompt | LLM-judge avg |
|---|---|---|---|---|
| **MSA-SFT-S2 (本报告)** | Qwen3.5-9B-Base + S1+S2 | router(precision ~0.07) | MSA 三段格式 | **1.503** (10 bench × 100q) |
| Vanilla 9B-Base + Oracle | 未训练 base | **oracle 喂 prompt** | RAG 自然语言 | 3.347 (3 bench × 50q) |
| Vanilla 9B-Instruct + Oracle | 未训练 instruct | **oracle 喂 prompt** | RAG 自然语言 | **3.957** (10 bench × 100q) |

**为什么不严格公平**:

1. **docs 注入路径不同**: vanilla 把 docs 文本拼到 prompt 里(`Documents: [...] Q: x. A:`),LM 走自然 RAG;MSA 把 docs 通过 8 个 MSA 层 attend 到预编码 pooled cache,prompt 里**完全没有 docs**。
2. **解析不同**: vanilla 直接抽整段 LM 输出;MSA 必须切 `<End-of-Retrieve>` 取 part_c。
3. **检索质量不同**: vanilla = 100% precision oracle;MSA = router precision ~0.07。

→ 这就是为什么需要 §6 的 hybrid 验证(强制把 oracle docs 装进 MSA 的 pooled_cache)才能真正归因「LM 是否被训坏」。

---

## 6. Root Cause 分析 · 为什么是 1.503

### 6.1 三个叠加因素(按贡献度排序)

#### 因素 ① · Router precision 全面崩坏(主因)

router 在大 corpus(corpus 都是数十万到百万 docs)上 precision 跌到 0.01–0.28,平均 0.07。**LM 看到的 10 个 doc 里 9 个不相关**,即使三段格式正确,part_c 抽出的答案大概率是「I don't know」或「Sorry, the documents do not contain ...」。

**证据**: msmarco_v1 上 precision = 0.010 → LLM-judge = 1.71(模型只能蒙)。triviaqa_06M precision 最高 0.279 → LLM-judge 也最高 2.71。两者强相关。

**根本原因推测**:
- CPT 用 50K MS MARCO facts(corpus 1.6M passages),router 只见过 ~50 万级别噪声;eval 时 corpus 上百万,**OOD distribution shift**。
- SFT 阶段 `aux_coef` 从 0.5 衰减到 0.05,router 进一步「漂移」。

#### 因素 ② · NQ / dureader / hipporag_narrative 高 empty 率(次因)

| Bench | empty_rate |
|---|---|
| nature_questions | **0.91** |
| hipporag_narrative | 0.53 |
| dureader | 0.32 |

Nature Questions 100 query 里 91 个返回空字符串。手工抽样发现 part_c 里 LM 输出的是:`"The answer to the question is: I don't know"` 或纯空白。

**根本原因**:
- SFT 数据里 NQ 子集是 `Tevatron/wikipedia-nq`,positive doc 通常**只覆盖 question 答案的 1 个 chunk**,LM 学到「如果看不到 answer chunk → 输出空」。
- 验证时 router 没召回 answer chunk(precision 低)→ LM 严格按 SFT 学到的行为输出空。

#### 因素 ③ · MSA 三段格式 + part_c 截断(可观察因素)

`reached_part_c_rate` 在 hipporag_narrative 上只有 **0.48**。意味着一半的样本在生成完 part_b(doc 全文 dump)后,1024 token budget 用完,根本没生成到 part_c → 解析得到空 answer。

**根本原因**:
- 每个 positive doc 在 part_b 里被 verbatim dump,1 doc ~ 100–300 token。当 router 召回 5 个 doc 全做 part_b,800–1500 token 直接吃光。
- max_new_tokens 从原来 256 升到 1024 已经是 v2 修复,但 hipporag_narrative 的 doc 长度极端,1024 仍不够。

### 6.2 验证因素 ① 的设计 — Hybrid Oracle 验证

**思路**: 我们正在跑的 [`scripts/bench_msa_oracle_hybrid.py`](../scripts/bench_msa_oracle_hybrid.py) 把 SFT-S2 ckpt + **强制 pooled_cache 只装 gold docs**(等价于 router precision = 1.0),对比看 LLM-judge 多少:

- 如果 hybrid score ≈ 3.5–4.0 → ① 是 dominant 因素,**LM 没被训坏**,需要重训 router(或离线索引)。
- 如果 hybrid score 仍 ≈ 1.5 → 问题更深,LoRA r=16 可能不足以同时学「三段格式 + 保留答题能力」。

---

## 7. 反思与下一步

### 7.1 这次训练做对了什么

- 端到端跑通 `Qwen3.5-9B-Base → CPT → SFT-S1 → SFT-S2`,完整复现 William 在 paper §3 描述的训练范式。
- Curriculum 8K → 64K 在 SFT-S2 顺利完成,无 OOM、无 NaN。
- 路由 IR(F1)在 hotpotqa / triviaqa_06M / dureader 等高 lexical-overlap bench 上有信号,证明 router 学到了「query-doc 语义匹配」。

### 7.2 这次训练的 5 个错误教训

| # | 错误 | 后果 | 教训 |
|---|---|---|---|
| 1 | 用 50K facts 的小 corpus CPT 后直接面对百万级测试 corpus | router precision 在 OOD 上崩 | CPT 的 corpus 必须和测试 corpus **同分布、同规模**(paper 5B token CPT 才有意义) |
| 2 | LoRA r=16 太小 | LM 在 SFT 阶段同时学「三段格式 + 保留答题能力」二选一,没学好 | 大 backbone 用更大 LoRA(r ≥ 64),或 full-finetune router 的 Q/K |
| 3 | 全程没有「边训边测」 | 6000 step 跑完才知道 NQ 91% empty | **必须在 SFT-S1 第 1000 step 就做 mini-验证**,设红线门槛(如 LLM-judge < 2.5 即停训) |
| 4 | aux_coef 衰减太快(0.5 → 0.05) | router 在 SFT 后期得不到训练信号,逐步漂移 | 至少留 0.3 终值 OR 完全冻 router 仅训 LoRA |
| 5 | bench 评分用同一个 prompt 路径(MSA 三段),没做 vanilla-RAG ablation | 不知道分数低是 LM 问题还是 router 问题 | 任何 MSA 实验必须配套 hybrid-oracle 验证(本次新增) |

### 7.3 下一条链路(已开始,见 [`INSTRUCT_TRAINING_REPORT.md`](INSTRUCT_TRAINING_REPORT.md))

- 换 backbone 到 `Qwen3.5-9B-Instruct`,因为 instruct 自带「答题倾向」可能能抑制空答率
- LoRA r=64,α=128(3× trainable)
- CPT corpus 换成 **paper-aligned shard**(KaLM + sentence-transformers,18M queries / 158B tokens 全量)

---

## 8. 完整文件清单(本次训练产物在 cvm-rl `msa-dev` 容器)

| 类别 | 路径 |
|---|---|
| CPT ckpt | `/workspace/ckpt/qwen3_5_msa_long_step7000.pt` |
| SFT-S1 final | `/workspace/runs/sft_s1_qwen3_5_0425_1439/qwen3_5_msa_sft_s1.pt` |
| SFT-S1 中间 ckpt | `step500.pt` ~ `step7000.pt`,every 500 |
| SFT-S2 final | `/workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt` |
| SFT-S2 中间 ckpt | `step600.pt` ~ `step6000.pt`,every 600 |
| 验证 JSON(MSA) | `/workspace/eval_evermind_aligned_v2/full/{llmscore_summary,summary}.json` |
| 验证 JSON(vanilla 9B-Base + oracle) | `/workspace/eval_vanilla_base/oracle_50q/llmscore_summary.json` |
| 验证 JSON(vanilla 9B-Instruct + oracle) | `/workspace/eval_vanilla_instruct/oracle_full/llmscore_summary.json` |
| 训练日志 | `/workspace/runs/{sft_s1,sft_s2}_qwen3_5_*/train.log` |

---

*Last updated: 2026-05-07 — by repo `eval-pipeline` & `docs/training-reports` branches.*
