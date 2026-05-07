# MSA Paper Alignment Audit

> **目的**：在下一轮重训之前，把 MSA 论文（arXiv:2603.23516）+ 官方推理仓库（[EverMind-AI/MSA](https://github.com/EverMind-AI/MSA)）跟我们当前训练实现做一次完整对照，固化所有 hyperparameter / 训练范式 / 数据格式上的偏差，作为下一轮训练的 ground-truth blueprint。
>
> 这份 audit 由前两次失败的训练（base 链 LLM-judge 1.503/5、instruct 链 5b CPT 损失 flat）反推得出。

---

## 0. Executive Summary（结论先行）

| 维度 | Paper / 官方 | William msa-minimind | 我们 base 链 | 我们 instruct 链 |
| --- | --- | --- | --- | --- |
| **Backbone** | Qwen3-4B-Instruct-2507 | MiniMind 64M | **Qwen3.5-9B-Base** ❌ | Qwen3.5-9B-Instruct ✅ |
| **CPT corpus** | 158.95B tokens, deduplicated | MS MARCO v2.1 ~0.82B | MS MARCO v2.1 50K facts ≈ **35.84M** ❌ | KaLM + ST mix shard ≈ **696M** ⚠ |
| **CPT 比例** | 1.0× | 1/200 | **1/4400** | 1/200 |
| **chunk size P** | 64 | 64 | **32** ❌ | 64 ✅ |
| **top-k** | 16 | 16 | **8** ❌ | 16 ✅ |
| **router_layer_idx** | 上半层 | layer 4–7 of 8 | layer 18–35 of 36 ✅ | layer 18–35 of 36 ✅ |
| **decouple_router** | 是（KV-head GQA） | 是 | 是 ✅ | 是 ✅ |
| **aux_loss_method** | InfoNCE Eq.(5)，τ 未明说 | InfoNCE，τ=0.07 | INFONCE_DECOUPLE，τ=0.1 ✅ | INFONCE_DECOUPLE，τ=0.1 ✅ |
| **Warmup** | L=0.1·LM+L_aux, lr=1e-4 | 同 paper | warmup 1000 step, lr=1e-4 ⚠ | warmup 2000 step, lr=1e-4 ✅ |
| **Main** | L=LM+0.1·L_aux, lr=6e-6 | 同 paper | main 6000 step, lr=**1e-5** ⚠ | main, lr=6e-6 ✅ |
| **训练范式** | **multi-round interleave**：模型只生成 doc-id 和 final answer，doc 原文由系统注入 | **single-pass three-part**：模型一口气生成 doc-id + doc text + answer | 同 William ❌ | 同 William ❌ |
| **SFT 数据切分** | 多跳 chain 切成多个 single-step 样本 | 已实现 split_multi_hop | 用 sft_mix 5 benchmark | 用 sft_mix 5 benchmark |
| **SFT-S1 context** | 8K | 8K（短） | ~8K ✅ | ~8K ✅ |
| **SFT-S2 context** | 8K → 64K curriculum | 9K → 65K | 9K → 65K ✅ | 未跑完 |
| **Inference template** | QWEN3_INSTRUCT_TEMPLATE | 自定义 build_msa_train_prompt | 同 William ❌ | 同 William ❌ |

**最关键的 3 个偏差（按修复优先级）：**

1. **训练范式不匹配 paper（P0）**：paper / 官方推理是 **multi-round interleave**，模型每轮只生成 doc-id 或 final answer；William / 我们是 **single-pass**，模型在一个 forward 里生成 doc-id + 完整 doc 原文 + answer。结果：part_b（doc 原文）吃掉绝大多数 SFT 容量和 max_new_tokens budget → part_c（answer）截断 → **91% empty answer**。这是 base 链 1.503 分的最可能根因。
2. **base 链 backbone 选错（P0）**：paper 明确从 Instruct-2507 起步；我们 base 链拿了 Qwen3.5-9B-**Base** → 模型完全没有 chat / instruction-following 先验，要 SFT 才能学会，而我们的 SFT 容量都被 part_b 吃掉了。
3. **base 链关键 MSA hyperparam 缩水（P1）**：chunk=32 vs paper 64；top_k=8 vs paper 16；num_docs=32 vs William 64 vs paper 全 corpus。这把训练时的稀疏度跟推理时的稀疏度调成了不同区间，router 学的相似度分布在推理时偏移。

---

## 1. Paper §3.3.1 — Continuous Pre-Training

### 数据 & 规模
- Corpus: **158.95 B tokens**（deduplicated），paper 没公布具体组成
- Objective: **Generative Retrieval** — 模型自回归输出 unique document ID
- Backbone init: **Qwen3-4B-Instruct-2507 official weights**, router projector 随机初始化

### 损失
- `L_LLM`：标准 cross-entropy on target sequence
- `L_aux`：Eq.(5) supervised contrastive InfoNCE，对每条 query q：
  ```
  L_aux = -1/|P| · Σ_{i in P} log [ exp(s+_i / τ) / (exp(s+_i / τ) + Σ_{j in N} exp(s−_{i,j} / τ)) ]
  ```
  τ 在 paper 里没写具体值，**官方代码用 INFONCE_DECOUPLE_FOCAL，τ=0.1**

### Two-phase schedule
| 阶段 | Loss | LR |
| --- | --- | --- |
| Warmup | `L = 0.1·L_LLM + 1.0·L_aux` | **1e-4** |
| Main | `L = 1.0·L_LLM + 0.1·L_aux` | **6e-6**（cosine annealed） |

### MSA hyperparam（§4.1 ImplementationDetails）
- compression chunk size: **64** tokens
- top-k: **16** documents
- router 仅施加于上半层（官方 `ROUTER_LAYER_IDX="18,19,...,35"` for 36-layer model）
- router_q_proj：`hidden → num_attention_heads × head_dim`
- router_k_proj：`hidden → num_key_value_heads × head_dim`（**GQA aligned**）
- decouple_router=true, head_reduce=mean, query_reduce=max, chunk_reduce=max, decouple_pooling=mean

---

## 2. Paper §3.3.2 — Post-Training (SFT)

### Stage 1（MSA-S1）
- Context length: **8K tokens**
- 任务: SFT on QA datasets，建立 instruction-following + reasoning
- 数据规模: paper 没明说，但说"large-scale dataset"

### Stage 2（MSA-S2）
- 在 S1 之上 continue
- **数据清洗**：过滤错误 / 低质量样本
- Memory context 从 **8K → 64K**（curriculum extension）
- 目的：让模型外推到推理时的大 memory bank（277K → 10M tokens）

### Multi-hop 切分（§3.5 末尾）
> "each retrieval chain in the multi-hop datasets is divided into multiple training samples during model training. Each sample contains a single retrieval step, either based on the single query or on the existing document context, and samples are randomly selected for training."

→ HotpotQA / MuSiQue / 2Wiki 的多跳链在训练时被拆成单步样本，每个样本要么基于原 query、要么基于已有 doc context 再 retrieve 一次。

---

## 3. Paper §3.4 — Three-Stage Inference Process

### Stage 1: Global Memory Encoding（offline，一次性）
对整个 corpus 做 forward → 缓存 chunk-pooled `(K̄, V̄, K̄_R)`

### Stage 2: Routing & Context Assembly（online）
1. query → 计算 hidden states
2. router_q_proj → `Q_R`
3. `Q_R` 跟 cached `K̄_R` 算余弦相似度 → top-k
4. 仅 load top-k 的 `(K̄, V̄)`，跟 query local `K_q, V_q` concat

### Stage 3: Sparse Generation（online，autoregressive）
- 在 `[{K̄_topk}; K_q]` 上自回归
- **生成 `[id1] [id2] ... [idK] <End-of-Retrieve>` 后，遇到 delimiter → 系统查 doc 原文 → inject 到 input → 继续生成 final answer**
- Memory Interleave：可多轮 retrieve → context expand → 再 retrieve → ... → 最终 answer

### Memory Parallel（§3.4.2）
- `K̄_R` 分片到多 GPU（GPU-resident routing keys）
- `K̄, V̄` 存 CPU DRAM（CPU-offloaded content KVs）
- top-k 选中后异步 fetch 到 GPU
- 100M token 推理在 **2× A800** 实现

---

## 4. EverMind-AI/MSA 官方推理代码关键细节

### `scripts/resave_model.sh` — 模型 export 配置
```bash
POOLING_KERNEL_SIZE=64
TOP_K_DOCS=16
ROUTER_LAYER_IDX="18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35"

LMLOSS_WEIGHT=1.0
REC_LOSS_WEIGHT=0.0
AUX_LOSS_WEIGHT=0.1
ANS_LOSS_WEIGHT=1.0
AUX_LOSS_METHOD="INFONCE_DECOUPLE"
INFONCE_LOSS_TEMP=0.1

DECOUPLE_ROUTER="true"
HEAD_REDUCE_METHOD="mean"
QUERY_REDUCE_METHOD="max"
CHUNK_REDUCE_METHOD="max"
DECOUPLE_POOLING_MODE="mean"
```

### `scripts/run_benchmarks.sh` — 推理 benchmark 配置
- **8 GPU**，11 个 benchmark
- `top_p=0.9, temperature=0.0, max_length=2048`
- `template=QWEN3_INSTRUCT_TEMPLATE`
- `max_chunk_per_block=16384, block_size=2048`

### `src/msa/generate.py` — multi-stage interleave generate（**关键**）
**这是跟 William 实现最大的差异点**。官方 generate 是一个 multi-round 循环：

```
generate_stage = 1
while not finished:
    forward → next_token
    response_string += next_token

    if generate_stage in {2, 3}:                      # transition
        if '<End-of-Retrieve>' in inner_string:
            # Stage 3 transition
            inject "<|im_start|>The user's question is: {Q}<|object_ref_end|>"
        else:
            # Stage 2 transition
            doc_ids = parse [id]+ from inner_string
            inject "[id1]. {doc_text1}\n[id2]. {doc_text2}\n...<|object_ref_end|>"
        re-prefill with injected source_context
        continue

    elif token == '<|object_ref_end|>':
        generate_stage = 2  # next iter will inject doc text
    elif token == '<End-of-Retrieve>':
        generate_stage = 3  # next iter will inject question prompt
```

→ 模型在 Stage 1 只生成 doc id 字符串，Stage 2 transition **不是模型生成的**，而是系统从 corpus 查到 doc 原文后 inject 进 input 重新 prefill。模型只生成 final answer。

### 推理依赖
- torch 2.6, transformers **4.51.3**, **flash-attn 2.7.4.post1**, liger_kernel 0.5.10
- 必须用 flash_attn_varlen_func（官方 attention 实现强依赖）

---

## 5. William msa-minimind 复现的设计选择

> 根据 `/home/zeyu/msa/SUMMARY_CN.md` 和 `/home/zeyu/msa/PLAN_CN.md`。

### 训练范式 — single-pass three-part
William 把 paper 的 multi-round interleave **拍扁成 single sequence 训练**：

```
prompt (no loss):
  <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n

target (full SFT loss):
  Part A: [gid1] [gid2] ... [gidK]<|object_ref_end|>\n
  Part B: [gid1]. {original_doc_text_1}<|object_ref_end|>\n
          [gid2]. {original_doc_text_2}<|object_ref_end|>\n
          ...
  Part C: <End-of-Retrieve>\n{answer}<|im_end|>
```

**理由**：避免实现 multi-stage generate 的复杂度，让 SFT 信号直接对齐生成目标。

**问题**：
1. Part B 是 **逐 token 复述 doc 原文**，N 篇 × max_doc_len = 64 × 256 = 16384 tokens 全部走 LM loss
2. SFT 容量被 Part B 吃光，Part C answer 学得弱
3. 推理时 max_new_tokens 也被 Part B 吃光 → answer 截断 → empty
4. 跟官方推理 pipeline（`src/msa/generate.py`）不兼容 → 没法用 EverMind-AI/MSA 的 inference engine 加载我们的 ckpt

### 已对齐 paper 的部分（William SUMMARY_CN.md §326-347 表格）
- ✅ 架构：MSAConfig/Attention/Block/Model 全部按 paper Eq.(2)/(5) 实现
- ✅ Doc-wise RoPE + global RoPE query offset by k
- ✅ router_k_proj 用 num_kv_heads（GQA），跟官方一致（paper 没明说）
- ✅ Eq.(5) batched supervised InfoNCE，multi-positive
- ✅ Two-phase schedule + Generative Retrieval target 包含 doc 原文注入
- ⚠️ Backbone：用 MiniMind 64M（compute-bound 替代 Qwen3-4B）
- ⚠️ CPT 规模：MS MARCO v2.1 ~0.82B unique tokens vs paper 158.95B（1/200）

---

## 6. 我们 base 链 / instruct 链的实际配置

### base 链（已结束，LLM-judge 1.503/5）

| 项 | 配置 | vs paper / William |
| --- | --- | --- |
| Backbone | Qwen3.5-9B-Base | ❌ paper 是 Instruct-2507 |
| CPT data | MS MARCO v2.1 50K facts | ≈ 35.84M tokens（1/4400 paper, 1/23 William） |
| CPT steps | 7000 | William 35400（5×） |
| CPT batch | 1 | William 8 |
| CPT lr (main) | **1e-5** | paper 6e-6 |
| LoRA | r=16, alpha=32 | paper full-finetune |
| chunk_size | **32** | paper 64 |
| top_k | **8** | paper 16 |
| num_docs | 32 | William 64 |
| max_doc_len | 160 | William 256 |
| max_query_len | 192 | William 256 |
| SFT-S1 | 7000 step, ~35M tokens | OK |
| SFT-S2 | 6000 step, ~213M tokens, curriculum 64→512 docs | OK |
| 训练范式 | single-pass three-part | 同 William ❌ paper |
| 最终 LLM-judge | **1.503/5** avg | paper MSA-4B-S2 = 3.760 |

### instruct 链（部分跑完，未 evaluate）

| 项 | 配置 | vs paper |
| --- | --- | --- |
| Backbone | Qwen3.5-9B-Instruct | ⚠ paper 是 4B-Instruct-2507 |
| Phase 5a CPT | 51K step, ~437M tokens | 1/360 paper |
| Phase 5b CPT | 30K step, ~259M tokens（loss flat 16K step → killed） | — |
| LoRA | r=64, alpha=128 | paper full-finetune |
| chunk_size | 64 ✅ | |
| top_k | 16 ✅ | |
| Fork SFT-S1 smoke | 1000 step, ~5M tokens → 100% empty | 容量严重不足 |
| Fork SFT-S2 | killed @ step 25 | — |
| 训练范式 | 同 base 链 single-pass | 同 William ❌ paper |
| 最终 LLM-judge | 未 evaluate | — |

---

## 7. 关键差异表 + Root Cause Hypothesis

| # | 偏差 | 影响 | 优先级 | 建议修复 |
| --- | --- | --- | --- | --- |
| 1 | **训练范式 single-pass vs paper multi-round interleave** | Part B 吃光 SFT 容量；推理时 answer 截断（91% empty on NQ） | **P0** | 改训练数据为 multi-round 格式（每个 multi-hop chain 拆成 2-3 个 sample），target 只含 doc-id 序列 *或* final answer，绝不让模型复述 doc 原文 |
| 2 | **base 链用 Qwen-Base 没 chat 先验** | 模型完全无 instruction-following，SFT 又被 Part B 浪费 → 模型连 answer 风格都没学会 | **P0** | 后续都从 Instruct chain 走 |
| 3 | **base 链 chunk=32, top_k=8** vs paper 64/16 | 训练时稀疏度跟推理不一致，router 学的相似度分布偏移 | P1 | 严格用 chunk=64, top_k=16（instruct 链已 OK） |
| 4 | **CPT 规模仅为 paper 1/200 ~ 1/4400** | LM loss 没收敛到 paper 水平（paper 应该 < 2.0），router 在 ID 生成上学得不深 | P1 | instruct 链至少跑到 1B+ tokens；考虑 LR warmup 后期重启 |
| 5 | **LoRA vs full-finetune** | paper 是 full-finetune，我们用 LoRA 限制了 backbone 的 representational change，可能让 MSA layer 难以收敛 | P2 | 第一轮重试可继续 LoRA r=64 + alpha=128，但若仍失败考虑 full-finetune backbone 后半层 |
| 6 | **没装 flash_attn** | 训练慢 2-5×（doc-encoding 全 padded SDPA），但不影响最终精度 | P2 | 装 flash-attn==2.7.4.post1 |
| 7 | **eval 用 William single-pass parser** | 跟 paper 推理不兼容，但跟我们训练匹配（自洽） | P2 | 重训后 eval 必须保持自洽（要么改 generate 用 multi-round，要么 stays single-pass） |
| 8 | **eval API key 失效** | 当前 hybrid + vanilla BM25 都跑完了 telemetry，但 LLM-judge 401 User-not-found | P0 | user 提供新 OPENROUTER_API_KEY 即可恢复 |

---

## 8. 当前 apple-to-apple eval 状态（截止文档撰写时）

### Hybrid-Oracle Eval（cvm-rl, `bench_msa_oracle_hybrid.py` v2）
**目的**：把 base 链 SFT-S2 ckpt 的 LM 能力跟 router 解耦 — 强行把 oracle gold doc 灌进 `pooled_cache`，绕过 router。看 LM 在"完美检索"下到底能不能输出 part_c。

**Smoke (1 bench × 10 q)**：
- musique: empty=0.10, reached_part_c=0.90, reached_im_end=0.90
- avg n_chars=1334, max=2478（说明 part_b 吃了大量 budget，跟 hypothesis #1 完全吻合）
- LLM-judge: API key 401，待恢复

**对比基线**：
- 同 ckpt + 真 router（base 链 final eval）：empty=0.91 on NQ, 0.03 on musique
- → oracle 注入让 musique 的 reach_C 从未知 → 0.90，**LM 本身有部分能力**，但 Part B 仍然吃 budget 导致部分 truncate

### Vanilla Instruct + BM25 RAG（william-dev, `bench_vanilla_instruct.py`）
**目的**：建立"训练前的 instruct 模型 + 真实 BM25 RAG"基线，看 paper 的 backbone-RAG 在我们的 Qwen3.5-9B-Instruct 上的水平。

**Smoke (3 bench × 30 q)**：
- musique BM25: empty=0%, F1=?
- hotpotqa BM25: F1=0.39, empty=0%
- nature_questions BM25: F1=0.31, empty=0%
- LLM-judge: API key 401，待恢复

**已知 oracle 基线（前期跑过）**: Qwen3.5-9B-Instruct + oracle = LLM-judge **3.957/5** avg

---

## 9. 下一轮重训蓝图（基于本 audit）

### Phase 0 — 修复评测工具链
- [ ] user 提供新 OPENROUTER_API_KEY
- [ ] 完成 hybrid-oracle + vanilla BM25 的 LLM-judge
- [ ] 把"vanilla Instruct + BM25"的 9 bench 全跑完，定标 paper Table 2 的 RAG R@1/5/10 baseline 在我们 9B 模型上的实际数值

### Phase 1 — 改训练范式（最关键，P0）
- [ ] 在 `msa/dataset_msa_sft.py` 里新增 multi-round mode：
  - target 只包含 `[id1] [id2] ... [idK]<End-of-Retrieve>\n{answer}<|im_end|>`
  - **完全去掉 Part B（doc 原文复述）**
  - 多跳样本拆 single-step（已实现 split_multi_hop，复用即可）
- [ ] 训练时 doc 内容从 `pooled_cache` 注入（沿用现有 sparse_generator 接口）
- [ ] 推理时也用 multi-round generate（参照官方 `src/msa/generate.py`）

### Phase 2 — 重启 instruct chain SFT
- [ ] Resume from `p5b_step20000.pt`（KaLM+ST mix 已 CPT 696M tokens）
- [ ] SFT-S1：5K step on sft_mix（new multi-round format），8K context
  - 监控：empty_rate < 5%（基于训练时的 mini-eval）
  - 监控：part_c 截断率 < 10%
- [ ] mini-eval（每 1000 step）on 1 bench × 30q：LLM-judge 必须 ≥ 2.5（不达标立刻停）
- [ ] SFT-S2：3K step curriculum 64→512 docs

### Phase 3 — 红线
若任一 mini-eval 满足以下任一条件，立刻 kill 训练：
- empty_answer_rate > 0.30
- LLM-judge < 2.0（vanilla Instruct + oracle 是 3.957，训完不能比 vanilla + oracle 还低）
- router top-1 hit-a-positive < 0.50
- LM loss 16K step 内变化 < 0.05（loss flat → kill，参考 5b 教训）

### Phase 4 — 全量 evaluation
- [ ] 9 个 paper benchmark 全跑（musique, hotpotqa, 2wiki, nq, ms_marco_v1, dureader, popqa, narrativeqa, triviaqa）
- [ ] LLM-judge 0-5 scale, judge model = `google/gemini-2.5-flash`
- [ ] 横向对比：
  - paper MSA-4B-S2: avg 3.760
  - vanilla Qwen3.5-9B-Instruct + oracle: 3.957
  - **目标**：我们的 MSA-9B-S2 ≥ 3.5（距 paper 7%，距 vanilla+oracle 12%）

---

## 10. 参考资料速查

| 资料 | 路径 | 关键章节 |
| --- | --- | --- |
| MSA paper PDF | `/tmp/evermind_msa/paper/MSA__Memory_Sparse_Attention_*.pdf` | §3.3, §3.4, §3.5, §4.1 |
| Paper 提取文本 | `/tmp/msa_paper.txt` | 全文 |
| 官方推理 repo | `/tmp/evermind_msa/` | `src/msa/`, `scripts/` |
| 官方 generate.py（multi-stage） | `/tmp/evermind_msa/src/msa/generate.py` | line 85-300 |
| 官方 attention | `/tmp/evermind_msa/src/msa/memory_sparse_attention.py` | INFONCE_DECOUPLE 实现 |
| 官方 hyperparam | `/tmp/evermind_msa/scripts/resave_model.sh` | 全部 export |
| William 复现总结 | `/home/zeyu/msa/SUMMARY_CN.md` | 14 步 + 对齐表 |
| 我们 base 链报告 | `docs/BASE_TRAINING_REPORT.md` | 全部 |
| 我们 instruct 链报告 | `docs/INSTRUCT_TRAINING_REPORT.md` | 全部 |

---

**Last updated**: 2026-04-24
**Author**: alignment audit triggered by user feedback "训练时要仔细参照论文，以及论文对应的 github repo，看看里面的代码"
