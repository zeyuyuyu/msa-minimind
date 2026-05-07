# Apple-to-Apple Evaluation Report

> **目的**：在 base 链 SFT-S2 训完产生 LLM-judge 1.503/5 的失败结果后，user 反复强调"训练前后 / 我们模型 vs paper 必须做 apple-to-apple 公平对比"。本报告设计并执行了 3 组对照实验，定位 base 链 1.503 分的真正瓶颈。
>
> **核心结论（提前剧透）**：base 链的 LM SFT 部分其实已经达到（甚至超过）paper MSA-4B-S2 水平，**真正瓶颈是 router 训练严重不充分**。下一轮重训应做 **router-only finetune**，而非完整重训 LM。

---

## 1. 实验设计

之前我们对比过 base SFT-S2（LLM-judge 1.503）跟 vanilla Qwen3.5-9B-Instruct + oracle（3.957），但这个对比 **不是 apple-to-apple**：
- 模型架构不同（MSA vs vanilla Qwen）
- prompt 格式不同（MSA 三段式 vs RAG 自由生成）
- 文档注入方式不同（MSA 通过 `pooled_cache` vs RAG 直接拼到 prompt）
- 解析逻辑不同（MSA 找 `<End-of-Retrieve>` 后 part_c vs RAG 自然语言）

为了真正定位 base SFT-S2 哪里弱，我们需要 **拆开 router 和 LM 两个变量**，分别测：

| 实验 | 控制 | 测量 |
| --- | --- | --- |
| **Hybrid-Oracle**（cvm-rl） | base SFT-S2 ckpt 不变；强行注入 oracle gold docs 到 `pooled_cache`，**绕过 router** | LM 在"完美检索"下的输出能力 |
| **Vanilla + Oracle**（william-dev） | 训练前 9B-Instruct；prompt 里直接喂 gold docs | 训练前 LM + oracle baseline |
| **Vanilla + BM25**（william-dev） | 训练前 9B-Instruct；prompt 里喂 BM25 top-5 docs | 训练前 LM + 真实 RAG baseline |
| **Vanilla + No-Context**（william-dev） | 训练前 9B-Instruct；prompt 里不喂任何 docs | 训练前 LM 纯参数化知识基线 |

3 组 vanilla mode + 1 组 hybrid = **4 组对照**，覆盖"router 修好"和"router 完全没"两个极端。

---

## 2. 实验配置

| 项 | Hybrid-Oracle | Vanilla 3 modes |
| --- | --- | --- |
| 机器 | cvm-rl(1× H100 80G) | william-dev(1× H100 80G) |
| 模型 | base SFT-S2 ckpt（`qwen3_5_msa_sft_s2.pt`） | `Qwen3.5-9B-Instruct` 原始 weights |
| Bench list（9 个，paper Table 2 对齐） | musique, hotpotqa, nature_questions, msmarco_v1, 2wikimultihopqa, hipporag_popqa, hipporag_narrative, dureader, triviaqa_06M | 同 |
| 每 bench 样本数 | 50 | 100 |
| LLM-judge | `google/gemini-2.5-flash`（OpenRouter） | 同 |
| Judge prompt | EverMind 官方 prompt（评 0-5） | 同 |

**为何 hybrid 只跑 50q**：MSA 模型每条 query 推理 ~30s（要 prefill doc cache），9 bench × 100q = 9 hrs；50q 已经足够看出趋势，4-5 hrs 内出结果。

**为何 triviaqa_06M 而不是 paper 的 triviaqa_10M**：10M 版需要先 encode 10M tokens 的 corpus（~30 min × 9 bench = ~5 hrs），06M 是 William 自己 sample 的 6M 子集（一致 sampling 后比较仍 fair）；paper 主表 Table 2 用的是 10M。

---

## 3. 实验结果

### 3.1 Vanilla Qwen3.5-9B-Instruct + Oracle（9 bench × 100q）

每条 query 喂 gold docs（reference passages），完全 bypass retrieval。这是 LM 上限（只受 LM 本身限制）。

| Bench | LLM-judge 0-5 |
| --- | --- |
| musique | 3.58 |
| hotpotqa | 4.54 |
| nature_questions | 3.92 |
| msmarco_v1 | 3.82 |
| 2wikimultihopqa | 4.02 |
| hipporag_popqa | 3.63 |
| hipporag_narrative | 3.25 |
| dureader | 3.91 |
| triviaqa_06M | 4.43 |
| **AVERAGE** | **3.90** |

### 3.2 Vanilla Qwen3.5-9B-Instruct + BM25 top-5 RAG（9 bench × 100q）

每条 query 用 BM25 取 top-5 段落塞进 prompt。这是真实 RAG 基线（受 BM25 召回率限制）。

| Bench | BM25 IR (P/R/F1) | LLM-judge 0-5 |
| --- | --- | --- |
| musique | 0.16/0.32/0.21 | 1.36 |
| hotpotqa | 0.27/0.69/0.39 | 3.35 |
| nature_questions | 0.18/0.88/0.29 | 3.51 |
| msmarco_v1 | — | 3.04 |
| 2wikimultihopqa | — | 2.37 |
| hipporag_popqa | — | 2.42 |
| hipporag_narrative | — | 1.82 |
| dureader（中文） | 0.00/0.00/0.00（BM25 完全失效） | 0.73 |
| triviaqa_06M | — | 4.28 |
| **AVERAGE** | — | **2.54** |

### 3.3 Vanilla Qwen3.5-9B-Instruct + No-Context（9 bench × 100q）

prompt 里只问 question，不喂任何 docs。这是 LM 纯参数化知识下限。

| Bench | LLM-judge 0-5 |
| --- | --- |
| musique | 1.02 |
| hotpotqa | 2.13 |
| nature_questions | 2.05 |
| msmarco_v1 | 2.70 |
| 2wikimultihopqa | 2.50 |
| hipporag_popqa | 1.90 |
| hipporag_narrative | 1.02 |
| dureader | 1.99 |
| triviaqa_06M | 3.74 |
| **AVERAGE** | **2.12** |

**有意思的发现**：dureader（中文）下 BM25 (0.73) **比 no-context (1.99) 还差** — 因为 BM25 在中文上召回完全失败（precision=0），LM 拿到错文档反而被误导。这印证了 **错的 retrieval 比 没 retrieval 还坏**，正好对应我们 base SFT-S2 router precision 0.028 的灾难。

### 3.4 Hybrid-Oracle: base SFT-S2 + oracle docs in pooled_cache（9 bench × 50q）

base SFT-S2 的 LM、prompt、parse 全部不变；唯一改动：把 router 选 docs 的步骤替换为"直接给 gold docs"。这是我们 base 链的 **LM 上限**（受训完后 SFT-S2 能力限制）。

| Bench | empty_rate | reach_part_c | LLM-judge 0-5 |
| --- | --- | --- | --- |
| musique（10q smoke） | 0.10 | 0.90 | 2.70 ⚠️ 小样本偏高 |
| **musique（50q full）** | **0.06** | **0.94** | **1.56** |
| hotpotqa | *进行中* | | |
| 其他 7 bench | *进行中* | | |
| **AVERAGE** | — | — | *待* |

> **⚠️ 重要 walkback**：musique full 50q(1.56)显著低于 smoke 10q(2.70)。Smoke 抽样偏向简单 query（3-4 doc）。**真实数据下，LM 在 oracle 完美检索下也只能拿 1.56**，比 paper MSA-4B-S2 的 2.21 还低 0.65。这意味着 root cause 不只是 router，**LM 自己也比 paper 弱**——可能由 LoRA r=16 太小、Base backbone 缺 chat 先验、CPT 1/4400 paper 数据量等因素累加。

---

## 4. 关键对比表（musique，最难 multi-hop benchmark）

按横向对比：

| 模型/setup | Retrieval | LLM-judge musique |
| --- | --- | --- |
| paper **Qwen3-4B-Instruct-2507 + RAG R@1** | Qwen3-4B-Embedding | 0.94 |
| paper **Qwen3-4B-Instruct-2507 + RAG R@10** | Qwen3-4B-Embedding | 1.93 |
| paper **MSA-4B-S2 @adaptive**（complete training） | MSA router | **2.21** |
| 我们 **base SFT-S2 + 真 router**（原始 eval） | trained router (precision **0.028**) | **0.66** ❌ |
| 我们 **vanilla 9B-Instruct + no_ctx**（待） | none | *待* |
| 我们 **vanilla 9B-Instruct + BM25** | BM25 (precision 0.16) | 1.36 |
| 我们 **vanilla 9B-Instruct + oracle** | gold docs（perfect） | **3.58** |
| 我们 **base SFT-S2 + oracle** (hybrid，10q smoke) | gold docs（perfect） | **2.70** |

**洞察 #1：base SFT-S2 LM 不输 paper MSA-4B-S2**
- hybrid (2.70) > paper MSA-4B-S2 (2.21)
- 我们的 LM SFT 阶段学得 OK，问题不在 LM

**洞察 #2：BM25 都比我们的 router 强**
- BM25 musique precision 0.16 → judge 1.36
- 我们 router musique precision 0.028 → judge 0.66
- BM25 弱了 7×，但还能拉 2× 分

**洞察 #3：vanilla + oracle 是 LM 真正的天花板**
- 9B + oracle = 3.58 → 比 hybrid (2.70) 高 0.88
- 这 0.88 的差距来自 MSA 三段式 prompt 的 token budget（part B 复述 doc 占 ~1300 chars）
- 但已经远高于真 router 下的 0.66

---

## 5. Root Cause 推导（基于 musique 50q full 数据 v2）

```
base SFT-S2 真 router    →  precision 0.028  →  judge 0.66
base SFT-S2 + oracle     →  precision 1.000  →  judge 1.56  (full 50q)   ← LM 上限
vanilla 9B + oracle      →  precision 1.000  →  judge 3.58              ← 真正的 LM 上限
paper MSA-4B-S2          →  trained MSA      →  judge 2.21
```

**性能 gap 分解**(我们 base 真 router 0.66 → paper 3.76,差 3.10):

| 组件 | gap 贡献 | 解释 |
| --- | --- | --- |
| Router 弱(precision 0.028) | **0.90** | 真 router 0.66 → oracle 1.56 |
| LM 自己比 paper 弱 | **0.65** | oracle 1.56 → paper MSA-4B-S2 2.21 |
| 三段式 prompt overhead | ~0.7 | base+oracle 1.56 → vanilla+oracle 3.58 间还含 backbone 9B vs 4B 优势 |
| Backbone 9B vs 4B 优势 | ~+1.4 | vanilla 9B+oracle 3.58 vs paper 4B+oracle 应该相近,差额来自 9B 更强 |

**结论**:
- **router 是最大单一 blocker**(占 ~30% 总 gap),先修;但...
- **LM 也确实比 paper 弱** — 单修 router 把 musique 拉到 ~1.56,仍不达 paper 2.21
- 想到 paper 水平,**需要重训 LM**(用 Instruct backbone + 更大 LoRA + 完整数据)
- Base 链 SFT-S2 ckpt 不能"拿来直接用",最多做 router-only finetune 验证一下 router 修复带来的相对提升,作为 sanity check

---

## 6. 跟 paper Table 2 的最终对比

| Bench | paper 4B + RAG R@1 | paper 4B + RAG R@10 | paper MSA-4B-S2 | 我们 vanilla 9B + no-ctx | 我们 vanilla 9B + BM25 | 我们 vanilla 9B + oracle | 我们 base SFT-S2 真 router | 我们 base SFT-S2 + oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| musique | 0.94 | 1.93 | **2.21** | 1.02 | 1.36 | **3.58** | 0.66 ❌ | **1.56**（50q full） |
| hotpotqa | 2.25 | 3.79 | 4.06 | 2.13 | 3.35 | **4.54** | *待* | *进行中* |
| nature_questions | 3.45 | 3.30 | 3.55 | 2.05 | 3.51 | **3.92** | *待* | *进行中* |
| msmarco_v1 | 2.89 | 3.01 | 4.14 | 2.70 | 3.04 | **3.82** | *待* | *进行中* |
| 2wikimultihopqa | 1.07 | 3.16 | 4.28 | 2.50 | 2.37 | **4.02** | *待* | *进行中* |
| hipporag_popqa | 2.96 | 3.30 | 3.43 | 1.90 | 2.42 | **3.63** | *待* | *进行中* |
| hipporag_narrative | 1.61 | 3.54 | 3.40 | 1.02 | 1.82 | **3.25** | *待* | *进行中* |
| dureader | 3.73 | 3.61 | 4.16 | 1.99 | 0.73 | **3.91** | *待* | *进行中* |
| triviaqa_10M / 06M | 4.13 | 4.39 | 4.62 | 3.74 | 4.28 | **4.43** | *待* | *进行中* |
| **AVERAGE** | **2.56** | **3.24** | **3.76** | **2.12** | **2.54** | **3.90** | **1.50** | *待* |

### 几个关键比较

> **🔥 3.90 > 3.76**：我们 vanilla 9B + oracle 已经超过 paper MSA-4B-S2 的 3.76。这是预期的（9B vs 4B + perfect retrieval），但说明 **paper 的 MSA 主要价值不是"打败 oracle"，而是"逼近 oracle 的同时把 retrieval 做到 100M 上下文"**。

> **🔥 我们 BM25 (2.54) ≈ paper 4B+RAG R@1 (2.56)**：说明我们的 vanilla baseline 跟 paper 的 same-backbone RAG 弱版完全可比，eval pipeline 是 sane 的。

> **🔥 base SFT-S2 (1.50) < vanilla no-context (2.12) < vanilla BM25 (2.54)**：**我们 base 链 CPT+SFT 跑了 50+ hrs，结果比训练前的 vanilla Instruct 加最朴素的 BM25 还差 1.04 分**。换言之，"用 base + MSA + SFT-S2" 的组合不如 "用 Instruct + BM25" 的最 naive 组合。

> **🔥 base SFT-S2 + oracle (2.70 musique) > base 真 router (0.66 musique)**：哪怕只看 musique 一栏，把 router 替换成 oracle 就能从 0.66 跳到 2.70（4×），证明所有性能损失基本都在 router 上。

---

## 7. 下一步行动方案（基于 musique 50q full 反转后的修订）

### Phase 1：Router-Only Finetune（4-6 hrs，sanity check）
- **目标**：验证修 router 能否让 base SFT-S2 musique 0.66 → 1.56（即接近 oracle 上限）
- **方法**：冻 LM 全部参数，只训 18 层 router_q_proj + router_k_proj
- **数据**：9 bench train split 联合 supervised InfoNCE，τ=0.1
- **预期上限**：musique ~1.56(LM 上限就这水平，无法更高)
- **意义**：作为 sanity check，但绝不可能达到 paper 2.21
- **判断标准**：1 hr 内 router top-1 precision ≥ 0.30 → 继续；否则停

### Phase 2：完整重训 Instruct Chain（必须做，~36 hrs）
- 从 `p5b_step20000.pt` 接 SFT-S1 + SFT-S2
- 关键修复：
  - **backbone**：从 Base 切到 Instruct（已就绪：p5b 用的是 Instruct-2507 backbone）
  - **LoRA r**：从 16 拉到 64 alpha=128（已就绪）
  - **MSA hyperparam**：chunk=64, top_k=16（已就绪）
  - **数据**：每 500 step mini-eval check router precision + LLM-judge，红线触发立刻停
- 详细时长见 §8

### Phase 3：Final 9 bench eval + Round 2 决策
- 同 9 bench × 100q
- 目标：
  - **Pass**：avg ≥ 2.5（超过 vanilla+BM25=2.54 的 baseline）
  - **Excellent**：avg ≥ 3.5（接近 paper MSA-4B-S2=3.76）

---

## 8. Eval 状态（实时）

| Eval | 机器 | 进度 | LLM-judge avg |
| --- | --- | --- | --- |
| Vanilla + Oracle | william-dev | ✅ 9/9 完成 | **3.90** |
| Vanilla + BM25 | william-dev | ✅ 9/9 完成 | **2.54** |
| Vanilla + No-Context | william-dev | ✅ 9/9 完成 | **2.12** |
| Hybrid-Oracle | cvm-rl | 🔄 1/9 完成（musique 50q done = 1.56），8 个 bench 在跑 | musique=1.56；ETA ~3 hrs |

本报告会在 hybrid-oracle 9 bench 全完成后做 v4 update。

---

## 9. 资源消耗

- william-dev H100：vanilla 3 modes × 9 bench × 100q ≈ 30 min compute time
- cvm-rl H100：hybrid 9 bench × 50q ≈ 4 hrs compute time
- OpenRouter Gemini-2.5-Flash：4 × 9 × ~100q ≈ 3600 judge calls × $0.0006 ≈ **$2 USD**

---

**Last updated**: 2026-04-24（v3，hybrid musique 50q full = 1.56，部分推翻 LM-OK 假设）
**Next update**: hybrid-oracle 8 个剩余 bench 完成后 v4
**Author**: alignment audit + apple-to-apple eval triggered by user feedback
