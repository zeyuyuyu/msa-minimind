# MSA × MiniMind 完整实现计划

> 目标：在 MiniMind-3 backbone 上，完整复现 MSA 论文（arXiv:2603.23516）描述的
> 全流程：**CPT 持续预训练 → 两阶段 SFT → 三阶段推理 → 9 个 QA benchmark 评测
> + NIAH 评测**。本文档分为「已完成」「下一步」「完整路线图」三部分。

---

## 第一部分：已完成工作

### 1.1 基础设施

| 编号 | 内容 | 产物 |
| ---: | --- | --- |
| 1 | UV 环境（torch-cu124 + transformers + datasets） | `pyproject.toml` |
| 2 | MiniMind 仓库克隆 & 预训练权重下载 | `minimind/out/pretrain_768.pth` (132 MB) |
| 3 | MS MARCO v1.1 / v2.1 数据下载 + MiniMind t2t_mini 语料 | `minimind/dataset/pretrain_t2t_mini.jsonl` (1.2 GB) |

### 1.2 MSA 模型实现

| 编号 | 内容 | 文件 |
| ---: | --- | --- |
| 4 | `MSAConfig` + `MSAAttention` + `MSABlock` + `MSAModel` + `MSAForCausalLM` | `msa/model_msa.py` |
| 5 | 解耦 router 投影（`qr_proj` 用 num_attention_heads，`kr_proj` 用 num_key_value_heads，GQA 对齐） | 同上 |
| 6 | Doc-wise RoPE + global RoPE（query offset by k） | 同上 |
| 7 | Eq.(2) 路由（max-token, mean-head, max-chunk cos）+ top-k 选择 + 稀疏注意力 | 同上 |
| 8 | Eq.(5) supervised InfoNCE 辅助损失（batched, multi-positive） | 同上 |
| 9 | `load_minimind_pretrained` — 完整加载 91 个 backbone 张量，router 保留随机初始化 | 同上 |

### 1.3 数据集

| 编号 | 内容 | 文件 |
| ---: | --- | --- |
| 10 | `Corpus` 类（持久化 global doc-ID 映射） | `msa/dataset_msa.py` |
| 11 | `MSACPTDataset` — 统一接口输出 `{doc_input_ids, pos_doc_labels, query_input_ids, labels}` | 同上 |
| 12 | 三个 builder：`build_msmarco_dataset` / `build_from_t2t_mini` / `build_synthetic_dataset` | 同上 |
| 13 | 目标格式对齐 paper Fig.3：`[id]<|object_ref_end|>` + `[id]. <text><|object_ref_end|>` + `<End-of-Retrieve>` + answer | 同上 |

### 1.4 训练

| 编号 | 内容 | 文件 |
| ---: | --- | --- |
| 14 | 两阶段调度（Warmup `0.1·LM + 1.0·aux, 1e-4` → Main `1.0·LM + 0.1·aux, 6e-6`），cosine LR | `msa/train_msa_cpt.py` |
| 15 | 自动配置推断（从 checkpoint shape 反推 backbone 架构） | 同上 |
| 16 | tokens 计数器、周期性保存、`--max_train_seconds` wall-clock 保护 | 同上 |

### 1.5 已执行训练与评测

| 编号 | 内容 | 结果 |
| ---: | --- | --- |
| 17 | 4 小时 paper-aligned CPT（MS MARCO v2.1, num_docs=64, top-k=16, P=64） | 35.4K main steps / 4.98B tokens / LM 2.02 / aux 0.05 |
| 18 | Held-out router 评测（446 validation queries, N=64） | top-1 **99.8 %**, top-4 **100 %**, aux 0.068 |
| 19 | Scaling-curve 评测（N=8..1024, 2 K..256 K context） | post-CPT LM 从 2.05 → 2.04（-0.5 %），top-1 100→99 % |
| 20 | 最终权重 | `out/msa_cpt_paper.pth` + 18 个中间 checkpoint |

### 1.6 Benchmark 适配器（已设计，未完全测试）

| 编号 | 内容 | 状态 |
| ---: | --- | --- |
| 21 | `msa/benchmarks.py`：6 个 adapter（MS MARCO, HotpotQA, TriviaQA, NQ, MuSiQue, 多跳按 §3.5 拆单步） | 代码就绪，smoke-test 未跑 |

### 1.7 已识别的设计问题与限制

在自审中已列出 8 项（见 `SUMMARY_CN.md` 第 14 步后的补充），关键点：

- Scaling-curve 的 "pre-CPT" 不是 paper 的 vanilla baseline（是 MSA 架构 + 随机 router）
- top-16 在 N≤16 时是 trivially 100%（denominator artifact）
- Doc-ID 用整数字符串（如 `[92345]` → 7 tokens），paper 未明确
- 训练规模仅 paper 的 3.14 %，模型远未收敛

---

## 第二部分：距离 paper 完整实现还差什么

按 paper 章节梳理未完成的部分：

### 2.1 训练管线（§3.3）

#### ⬜ **SFT 阶段 1**（§3.3.2 First Stage）
- 在 8K context 上做 SFT，基于 QA 任务
- 建立基础 instruction-following + reasoning 能力
- **输入数据**：各 benchmark train split（已有 adapter），筛选适合 8K 的样本

#### ⬜ **SFT 阶段 2**（§3.3.2 Second Stage / Curriculum）
- 数据清洗，过滤低质量样本
- 把 memory context 从 8K 扩展到 64K
- 目标：让模型学会 extrapolate 到更大 memory bank（100M 推理的前提）

#### ⬜ **Memory Interleave 训练样本**（§3.5）
- 多跳 benchmark（HotpotQA, MuSiQue, 2Wiki）的检索链拆成多个单步样本
- ✅ 已在 `benchmarks.py` 里实现 `split_multi_hop`
- ⬜ 需要验证拆分正确性 + 跑实际训练

#### ⬜ **训练语料扩展**
- 当前仅 MS MARCO v2.1（~0.82B unique tokens / 3.14 % of paper）
- 联合训练接入其它 benchmark train splits
- 预估联合后有 ~3-5B unique tokens，达到 paper 的 3-5 %

### 2.2 推理管线（§3.4）

#### ⬜ **Stage 1: Global Memory Encoding**（离线）
- 对整个 corpus 做一次 forward pass，cache `(K̄, V̄, K̄_R)`
- 持久化到磁盘（或 CPU memory）
- 接口：`encode_corpus(model, corpus_texts) → K̄, V̄, K̄_R`

#### ⬜ **Stage 2: Online Routing + Context Assembly**
- 接收 query，计算 `Q_R`
- 与 cache 的 `K̄_R` 匹配，取 top-k
- 仅加载 top-k 对应的 `(K̄, V̄)`，与 query 的 local `Kq, Vq` 拼接
- 接口：`retrieve_and_assemble(query_hidden, cache) → Kctx, Vctx`

#### ⬜ **Stage 3: Sparse Generation**
- 自回归生成 doc-ID 序列 + `<|object_ref_end|>`
- 系统查回 doc 原文插入上下文
- 继续生成 `<End-of-Retrieve>` + answer
- 接口：`sparse_generate(query, cache) → generated_tokens`

#### ⬜ **KV Cache 压缩**（§3.4.2 Tiered Storage）
- 对 100M tokens corpus：`K̄_R` 约 56 GB，`K̄, V̄` 约 113 GB
- Routing keys 常驻 GPU，content KV 存 CPU DRAM，top-k 选中后异步 fetch 到 GPU
- 单卡场景可简化，但需要完整接口
- 接口：`TieredKVStore` 类

#### ⬜ **Memory Parallel**（§3.4.2 多 GPU 推理）
- `K̄_R` 分片到多 GPU，query broadcast → 各 GPU 本地打分 → 全局 reduce
- 单卡场景可跳过，但需保留接口以便扩展
- H200 单卡内存够放 100M tokens 的 K̄_R（140 GB），但 content KV 需要 CPU offload

#### ⬜ **Memory Interleave 推理**（§3.5）
- 多轮 retrieval：生成 IDs → 查原文 → 再生成 → ... → 最终 answer
- 模型自适应决定每轮 retrieve 几个 doc、总共几轮
- 接口：`interleave_generate(query, cache, max_rounds)`

### 2.3 评测（§4）

#### ⬜ **9 个 QA benchmark 评测**（paper Table 2/3）
- MS MARCO v1, Natural Questions, DuReader, TriviaQA (10M), NarrativeQA, PopQA, 2Wiki, HotpotQA, MuSiQue
- 指标：LLM judge（0-5 分）
- ⚠️ paper 用 GPT-4 级模型做 judge；我们需要替代方案（Qwen3-4B-Instruct 自评？或 exact match + ROUGE）
- Memory bank 大小按 paper 配置（277 K ~ 10 M tokens）

#### ⬜ **RULER NIAH 评测**（paper Fig. 4）
- 8 个子任务（SA1-3, MK1-3, MV, MQ）
- Context 长度 32K → 1M
- 指标：accuracy
- 数据集：`hsiehjackson/RULER` 或本地生成

#### ⬜ **Paper Fig. 1 "Context Degradation" 复刻**
- 当前 scaling-curve 评测已达 256 K；要到 100 M 需要：
  - 更大的 held-out corpus（或允许负例重复采样）
  - 更小的 per-sample doc（把 256 tokens 压到 64 tokens）
  - 内存与时间预算

### 2.4 架构增强

#### ⬜ **长 context 支持**
- MSA 推理要求 `max_position_embeddings` 能容纳 top_k + query（当前 4096 足够）
- 但 content K̄/V̄ 来自 doc-wise RoPE（`L_d` 范围内），无需扩展全局 RoPE
- ⚠️ 如果 SFT 阶段 2 把 query 扩到 64 K，需要 YaRN 或直接 extend RoPE

#### ⬜ **Flash Attention varlen**（官方实现用了这个）
- 当前用 padded SDPA，文档编码浪费算力
- 切换到 `flash_attn_varlen_func` 能显著提速（2-5×）
- 与官方 `EverMind-AI/MSA` 对齐

#### ⬜ **Chunking / Gradient checkpointing**
- 支持更大 N（2048+）训练，需要梯度检查点降低显存

### 2.5 工程 & 可用性

#### ⬜ **`generate()` 接口**
- 当前 `MSAForCausalLM` 继承 `GenerationMixin`，但没覆盖 `prepare_inputs_for_generation`
- 需实现 MSA-specific generation：维护 doc bank + query，支持流式 output
- 接入 HF `pipeline` 或 vllm

#### ⬜ **转换为 HF 标准格式**
- 保存 `config.json` + `pytorch_model.bin` 以便被 `AutoModel` 加载
- 参考 `scripts/convert_model.py`（MiniMind 仓库）

#### ⬜ **serving**
- 参考官方 `src/msa_service.py` 写一个轻量 FastAPI server
- 支持多 GPU 推理（虽然 64M 单卡就够）

---

## 第三部分：完整路线图（按优先级）

### 🎯 **Phase A：完成核心训练 → 可用模型**（~1-2 周）

1. **Benchmark adapter 烟雾测试**（1 天）
   - 验证 `msa/benchmarks.py` 的 6 个 adapter 都能跑通
   - Streaming 模式小规模测试

2. **联合 CPT 训练**（2-3 天）
   - 接入 MS MARCO + HotpotQA + TriviaQA + NQ + MuSiQue
   - 训练 12-24 小时（目标 ~10-20 B tokens）
   - 产物：`msa_cpt_multibench.pth`

3. **SFT 阶段 1（8K context）**（1 天）
   - 复用 CPT 的正例数据，改为 instruction-following 格式
   - LR 3e-5, 短训（几千步）

4. **SFT 阶段 2（64K context）**（2 天）
   - 数据清洗 + context 扩展
   - 可能需要 YaRN 或简单 rope_theta 调整

### 🎯 **Phase B：推理管线**（~1-2 周）

5. **3 阶段推理实现**（3-4 天）
   - Stage 1 离线编码：`encode_corpus.py`
   - Stage 2 在线路由：`retrieve.py`
   - Stage 3 稀疏生成：改造 `generate()`
   - 单元测试：encode-retrieve-generate 端到端

6. **KV Cache 压缩 + CPU offload**（2-3 天）
   - `TieredKVStore` 类
   - 单卡 100M tokens 验证（H200 + 主机 DRAM）

7. **Memory Interleave 推理**（2 天）
   - 多轮循环
   - 终止条件：模型自适应输出 `<End-of-Retrieve>`

### 🎯 **Phase C：评测**（~1-2 周）

8. **9 benchmark QA 评测**（3-5 天）
   - 9 个 benchmark 的 dev/test split 评测脚本
   - LLM judge 选项（用更大模型做评分，或回退到 EM/F1/ROUGE）
   - 复刻 paper Table 2 的格式

9. **RULER NIAH 评测**（2 天）
   - 生成 NIAH 测试样本（32K → 1M）
   - 复刻 paper Fig. 4

10. **100 M context 压力测试**（2 天）
    - 构造或扩充 corpus 到 100M tokens
    - 端到端 QA latency & accuracy

### 🎯 **Phase D：工程化**（~1 周）

11. **HF 模型格式导出 + push to hub**
12. **FastAPI serving**
13. **文档（README + 训练 recipe + 推理示例）**

### 🎯 **Phase E：修复已识别的限制**（贯穿 A-D）

- ⬜ 真正的 "vanilla" baseline（纯 MiniMind concat）补 scaling-curve
- ⬜ Passage-disjoint held-out（撕开 train/eval 的 passage 重叠）
- ⬜ Doc-ID 改用专用 token（可选，需改 tokenizer）
- ⬜ Flash attention varlen packing

---

## 时间表估算

| 阶段 | 最短路径 | 标准路径 | 含所有限制修复 |
| --- | ---: | ---: | ---: |
| Phase A (核心训练) | 5 天 | 10 天 | 14 天 |
| Phase B (推理管线) | 7 天 | 12 天 | 14 天 |
| Phase C (评测) | 5 天 | 10 天 | 14 天 |
| Phase D (工程化) | 3 天 | 5 天 | 7 天 |
| **合计** | **20 天** | **37 天** | **49 天** |

按「标准路径」节奏（单开发 / 单 H200），~5 周可达到一个完整的「MSA × MiniMind」研究级实现，能端到端跑通 CPT + SFT + 三阶段推理 + 9 benchmark 评测。

---

## 下一步建议

**如果目标是尽快看到 paper-style 的 QA benchmark 表现**：
→ 跳到 **Phase C step 8**（直接用当前 CPT 权重做单步 retrieval + generation，在 MS MARCO dev 上出分数）。
  这是 paper Table 2 的核心数字，1-2 天就能有第一个结果。

**如果目标是完整复现**：
→ 按 A → B → C → D 顺序推进。Phase A step 1-2（benchmark 扩展训练）是最有价值的下一步，
  因为当前 CPT 只见过 MS MARCO，模型在其它 benchmark 上几乎没泛化信号。

**如果目标是学术文章**：
→ 优先 Phase E 中的"真正 baseline"（普通 MiniMind concat），把 scaling-curve 图做成
  真正的「MSA vs vanilla」对比，这是 paper Fig. 1 的核心卖点。
