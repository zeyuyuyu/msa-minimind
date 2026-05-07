# 训练失败复盘 & 防止再犯 Checklist

> **背景**：base 链 SFT-S2 训完拿 LLM-judge **1.503/5**（远低于 paper MSA-4B-S2 的 2.21，甚至低于 vanilla 9B 不喂 docs 的 2.12）。Instruct 链 CPT-5b 跑 30k step 才发现 flat loss。Fork-S1 smoke 1k step 100% empty answer。本文档分类整理失败原因 + 下一轮如何避免。
>
> 配套文档：
> - 根因证据：[`APPLE_TO_APPLE_EVAL_REPORT.md`](./APPLE_TO_APPLE_EVAL_REPORT.md)
> - 跟 paper 偏离的全面 audit：[`PAPER_ALIGNMENT_AUDIT.md`](./PAPER_ALIGNMENT_AUDIT.md)
> - 当前状态 & 接手指南：[`HANDOVER.md`](./HANDOVER.md)

---

## TL;DR：7 大类问题、3 条最致命

按"对最终结果伤害程度"排序，**P0 = 致命，P1 = 大伤害，P2 = 中等，P3 = 工程效率**：

| # | 问题 | 优先级 | 防止措施 |
| --- | --- | --- | --- |
| 1 | **没有训练中 LLM-judge 红线**（训完才发现退化） | **P0** | 每 1000 step 跑 1 bench × 30q LLM-judge，< 1.98 立刻 kill |
| 2 | **Router precision 不监控**（训完才知道 0.028 ≈ random） | **P0** | 每 200 step 在 dev set 算 precision/recall，< 0.05 alarm |
| 3 | **LoRA r=16 太小** + **CPT 数据 1/4400 paper 量** | **P0** | 严格抄 paper config（r=64/alpha=128），CPT ≥ 200M token |
| 4 | Backbone 选错（Base vs paper Instruct-2507） | P1 | 任何脱离 paper 的决策必须 Day-1 audit |
| 5 | SFT-S2 curriculum 跳太快（64→512 一次跳 8x） | P1 | 64→128→256→512 平滑，每档 ≥ 1500 step |
| 6 | Eval 早期不公平（vs vanilla 不是 apple-to-apple） | P2 | Day-1 设计对照实验矩阵 |
| 7 | 没 wandb / launcher 不 commit / 训练不在 tmux | P3 | 工程基础设施 checklist |

**最致命的 3 条**：①没监控 → 训完才知道崩了 → ②router 没在学 + ③LM 容量不够。这三条解决一条都能把 1.503 拉上 2.5+。

---

## 1. 监控 / 红线缺失（P0，最致命）

### 1.1 没有训练中 LLM-judge mini-eval

**问题**：base 链 SFT-S2 跑了 6000 step（42 小时）才在 final eval 发现 LLM-judge 1.503，**整个训练过程对 final score 完全盲跑**。

**证据**：
- `runs/sft_s2_qwen3_5_0426_0349/train.log` 全程只 log 了 `loss`、`lr`、`tokens/s`，没有 LLM-judge 数字。
- 等 42 小时训完跑全 eval 才看到 1.503 → 此时投入已经沉没。

**根因**：训练脚本 `sft_stage2_qwen3_5.sh` 没有 mid-training eval hook，主因是 LLM-judge 涉及外部 API 调用（OpenRouter），早期觉得"训练时插 judge 太重"。

**防止**：
1. 训练脚本里加 `--mid_eval_every 1000` flag，每 1000 step 暂停训练，跑 `bench_evermind_aligned_v2.py --num_queries 30 --benches musique`，再 `llm_judge_evermind.py` 评分。
2. 单次 mini-eval 成本 ≈ 5 min compute + $0.05 OpenRouter，**远低于"训完才发现崩了"的代价**。
3. 阈值：分数 < **1.98**（vanilla no-ctx baseline 2.12 × 0.93）→ 自动 kill 训练 + 邮件 / Slack 报警。
4. 输出存到 `runs/<exp>/mid_eval_step{N}.json`，方便事后画 loss-vs-judge 曲线。

### 1.2 没有 router precision/recall 监控

**问题**：base 链 SFT-S2 训完才发现 router precision = **0.028**（musique gold ratio 0.04 → router 几乎完全 random），训中过程对 router 是否在学完全不知。

**证据**：
- 整个 SFT-S2 training log 没有 `router_precision` 字段。
- 只有 final eval 才在 `bench_evermind_aligned_v2.py` 里算了 precision/recall。

**根因**：MSA training 的 loss 目标是 LM 自己生成 part_a doc-id（teacher forcing），不直接 supervise router 的 attention 权重。所以 LM loss 在降不代表 router 在学。

**防止**：
1. 训练时每 200 step 在 dev set（10 query × 1 bench）跑一次 router-only forward，记录：
   - `router_precision@k=8` （选中的 8 个 doc 里几个是 gold）
   - `router_recall@k=8`   （所有 gold 里几个被选中）
2. 阈值：precision < **0.05** 持续 1000 step → alarm（router 没在学）。
3. **更激进**：考虑直接对 router attention 加辅助 loss（例如 contrastive loss with gold doc index），强制 router 学到 retrieval 信号。这是 paper 的 Section 3.3 第 2 段提到但 William 实现里跳过的。

### 1.3 没有 empty_answer_rate 监控

**问题**：base 链 SFT-S2 在 nature_questions 上 91% 输出 empty answer（`<End-of-Retrieve>` 后面没 token，或者被 max_new_tokens 截断），训中没察觉。

**证据**：
- `eval_evermind_aligned_v2/nature_questions.json` 里 91% 样本 `part_c` 字段为 ""。
- 直到 final eval 才算出来。

**根因**：part_b 复述太长（avg 1300+ char），用光了 `max_new_tokens` budget；同时模型没学会"何时停止 part_b 进入 part_c"的边界 token。

**防止**：
1. mini-eval 里直接算 empty_answer_rate（generated text 里有没有 `<End-of-Retrieve>` 后面 ≥ 5 token）。
2. 阈值：> **0.30** → alarm。
3. 治标：增大 `max_new_tokens`（从 1024 → 2048）。
4. 治本：改用 multi-round interleave 范式（part_b 由系统注入，不消耗模型 generation budget）。或者在 SFT 数据里把 part_b 截短到 200 char 以内。

### 1.4 CPT-5b flat loss 跑了 30k step 才察觉

**问题**：instruct 链 Phase 5b CPT，前 14k step loss 正常下降，从 14k 到 30k 完全 flat，**16k step 都没人发现**，最后被 user 要求 kill 让出 H200。

**证据**：`p5b_shard1.log` 中 step 14000-30000 的 loss 围绕 2.10 上下波动 ±0.02，没有任何下降趋势。

**根因**：肉眼看 log 不及时 + 没自动告警；学习率太大或数据 shard 顺序导致 plateau。

**防止**：
1. 自动检测：rolling 1k-step window 的 loss 标准差 < 0.05 且均值变化 < 0.01 持续 2k step → alarm。
2. learning rate scheduler 加 plateau-aware 逻辑：检测到 plateau 自动 LR ÷ 2。
3. **永远开 wandb**：肉眼看 log 不能信。

### 1.5 没有训练 dashboard

**问题**：所有指标靠 grep log 文件 + jq 拼，没有集中可视化，错过几次明显异常。

**防止**：
1. **wandb 必开**（cvm-rl 容器没装,要 `pip install wandb`）；项目名建议 `msa-9b-cpt-2026`，每个 run 用 `<chain>_<stage>_<seed>` 命名。
2. 必须 log 的 metric：`loss`、`lr`、`tokens/s`、`gpu_mem`、`router_precision`、`router_recall`、`empty_answer_rate`、`mid_eval_llm_judge`（每 1000 step 一个点）。
3. wandb alert 配 webhook 到 Slack。
4. 备选：tensorboard + custom HTML dashboard（如果 wandb 不能用）。

---

## 2. 超参 / 模型容量（P0，第二致命）

### 2.1 LoRA r=16 太小

**问题**：base 链全程 LoRA `r=16, alpha=32`，paper / instruct 链都用 `r=64, alpha=128`。9B 模型 + r=16 ≈ 4M 可训参数，根本撑不起 MSA 这种大改造。

**证据**：
- `scripts/long_cpt_qwen3_5.sh` 写死 `--lora_r 16`。
- hybrid-oracle 测出来 LM 在完美检索下 6-bench AVG 只有 2.71，比 vanilla（无 LoRA）no-ctx 2.12 强一点点 → LoRA 容量不够吸收 paper 的 MSA-attention 行为。

**防止**：
1. 严格抄 paper config，**r=64, alpha=128, dropout=0.05** 起步。
2. 如果 GPU 紧张优先省别的（batch size、grad accum），不要省 LoRA r。
3. 9B 模型上 r=64 大约 16M 可训参数，单卡 H200 完全 hold 住。

### 2.2 MSA chunk_size / top_k 缩水

**问题**：base 链 `chunk_size=32, top_k=8, num_docs=32`；paper `chunk=64, top_k=16, num_docs=全 corpus`；William fork 用的 `num_docs=64`。**训练期 sparsity 跟推理期 sparsity 不一致**，router 学的相似度分布在 inference 时偏移。

**证据**：[`PAPER_ALIGNMENT_AUDIT.md` §3](./PAPER_ALIGNMENT_AUDIT.md) 表格全列。

**防止**：
1. 严格按 paper：`chunk=64, top_k=16, num_docs` 训练时 ≥ 256，eval 时全 corpus。
2. **训练 / eval / paper 三者 sparsity 配置必须 1:1 对齐**，否则 router 输出分布漂移。
3. 改 chunk_size 必须重新生成 `encoded_corpora/`（cvm-rl 现存的是 chunk=32）→ **再 encode 一次约 6 hr × H200**，提前预算时间。

### 2.3 SFT-S2 curriculum 跳太快

**问题**：SFT-S2 curriculum 从 `num_docs=64` **直接跳到 512**（8x），中间没有过渡，模型在长 context 阶段 loss 巨振。

**证据**：`sft_s2_qwen3_5_0426_0349/train.log` step 3000 那个跳点有明显 loss spike。

**防止**：
1. **平滑 curriculum**：`64 → 128 → 256 → 512`，每档 **至少 1500 step**。
2. 每次升档前先验证：当前档的 router_precision 是否稳定 > 上一档（说明长 context 已学会），否则停留多 1000 step。
3. 升档时 LR 降一档（× 0.5），避免 long-context 训不稳。

### 2.4 CPT 数据量严重不足

**问题**：base 链 CPT 总共 35.84M token；paper 量级是数百 B token（差 ~4400×）。MSA-attention 是 inserted module，需要大量 token 学到 cross-doc retrieval。

**防止**：
1. CPT 至少 **200M+ token**（5-6× 我们 base 链）。
2. 数据来源严格按 paper：KaLM (10B) + ST-mix (5B subset) + MS MARCO（已有）。
3. **CPT 的 pass@1 验证**：每 5000 step 在 dev set 上跑一次 NIAH（needle-in-a-haystack），看 long-context retrieval 是否在涨。

---

## 3. 数据 / 任务设计（P1）

### 3.1 Backbone 选错

**问题**：base 链选了 `Qwen3.5-9B-Base`，paper 用的是 `Qwen3.5-9B-Instruct`（甚至 paper 强调要 `Instruct-2507`）。Base 缺 chat / instruction-following 先验，三段式 prompt 全靠 SFT 学，但 SFT 容量已被 part_b 吃光。

**证据**：[`APPLE_TO_APPLE_EVAL_REPORT.md` §3.4](./APPLE_TO_APPLE_EVAL_REPORT.md) hybrid-oracle 数据。

**防止**：
1. **Day-1 paper alignment audit 列出所有偏离 paper 的决策**，每条记录"为什么偏离"和"风险评估"。
2. backbone 这种结构性决策必须强对齐。
3. 如果非要换，先用小 chain（CPT 1k step + SFT-S1 500 step + smoke eval）证明换的版本能 work，再放大。

### 3.2 SFT 数据 mix 任务分布不均

**问题**：sft_mix 偏向 multi-hop 长答 QA（musique/hotpotqa 占 70%+），short-answer task（NQ/popqa）占比小。最终 short-answer bench 翻车（NQ hybrid-oracle 1.22 < vanilla no-ctx 2.05）。

**证据**：`msa/data/sft_mix.py` 的 weight 配置。

**防止**：
1. **SFT mix 比例按 eval bench 分布配比**：9 个 bench 大约 5 个长答 + 4 个短答 → SFT mix 应该 long:short ≈ 1:1。
2. 短答任务专门加 instruction "答案 ≤ 5 words"，让 LM 学会停止。
3. 训练前在 dev split 上确认每个 bench 的 mid-eval 都有数。

### 3.3 Bench train split 没用上 / 用错

**问题**：paper 的 SFT 把 9 bench 各自 train split 都加进去；我们 sft_mix 可能漏了一些 bench 或者用了 eval split（污染）。

**防止**：
1. 写一个 `scripts/audit_data_split.py`，逐 bench 验证：
   - eval 用的 query 不在 SFT train 里（hash 比对）
   - paper 的 train split 全部进了 SFT mix
2. 训练 Day 0 必须跑这个 audit + 留 log。

---

## 4. 训练范式 / Inference 对齐（P2）

### 4.1 Single-pass vs Multi-round Interleave

**问题**：William 的训练 / inference 是 single-pass（part_a/b/c 一次 forward 出），paper 是 multi-round interleave（每轮只输 doc-id，系统注入 doc text，再输 answer）。Single-pass 让 part_b 1300+ char 吃掉 SFT 容量和 inference budget。

**证据**：
- 官方 inference 在 `/workspace/evermind_msa/src/msa/generate.py`。
- 我们实现在 `msa/inference/sparse_generator.py`（single-pass）。

**评估**：原以为这是 P0，hybrid-oracle 数据后降级为 **P3 工程效率差异**——single-pass 不是 fatal，但确实让 max_new_tokens budget 紧张。

**防止**：
1. **同时实现两种范式**，对比哪个 final score 高，再选一个。
2. 如果坚持 single-pass，把 part_b 长度限制在 200 char 内（数据预处理时截）。

### 4.2 Eval 流程不公平

**问题**：第一波对比把"我们 MSA model 走三段式 prompt + pooled_cache" vs "vanilla 走 RAG prompt 拼 docs 进 prompt"，**两套 evaluation pipeline 完全不同**，对比没意义。

**防止**：
1. **Day-1 设计对照实验矩阵**（4 组 × 9 bench × 3 mode），固定下来才开始训练。
2. 评估每个 ckpt 时同时跑：(a) 我们 MSA pipeline (b) hybrid-oracle (c) vanilla baseline。三个数字一起看。
3. 任何"绝对分数"要求都必须先确认对比是 apple-to-apple。

---

## 5. 工程基础设施（P3，但是个个都坑）

| # | 问题 | 防止 |
| --- | --- | --- |
| 5.1 | 训练用 `docker exec ... &`，SSH 一断 job 死 | 训练全部 `tmux new -s <exp>` 起，`tmux ls` 必须有 active session 才能 push |
| 5.2 | launcher script (`launch_p5_*.sh / launch_fork_*.sh`) 没进 git，机器一炸全丢 | 所有 launcher 必须 commit 到 `scripts/`，写 README 说明各 launcher 用途 |
| 5.3 | SFT-S2 跑 42 小时，中间没存增量 ckpt | 每 1000 step 存 ckpt，旧的清理留 3 份；ckpt 加 metadata.json（step / loss / mini_eval） |
| 5.4 | OpenRouter API key 多次失效 / 401，judge 卡住 | 备 2-3 个 key 自动 fallback，judge 失败重试 3 次 |
| 5.5 | Docker 环境差异（cvm-rl py3.11+torch2.6 vs william-dev py3.12+torch2.9）| 写死 Dockerfile 进 repo，所有训练机用同一个 image |
| 5.6 | `[python] <defunct>` 僵尸进程吃不到 GPU 但占显存 | 每周清理一次：`pkill -9 -f "python.*<defunct>"` |
| 5.7 | shell quoting 嵌套（docker exec 嵌 bash -lc 嵌 python）多次炸 | 长 command 写成 `.sh` 文件 scp 过去，再 docker exec 执行 |
| 5.8 | OPENROUTER key / HF token 容易写到代码 / 文档里被 GitHub secret scan 拦 | 用 `.env` + 加 `.gitignore`；文档里只写 placeholder |

---

## 6. 流程 / 决策（P3）

### 6.1 没 scope guard，一上来就跑全流程

**问题**：直接按 paper 全流程（CPT + SFT-S1 + S2）开跑，跑了 65 小时才知道结果不行。

**防止**：
1. **Day-0 Smoke**：1 bench × 100 step CPT + 100 step SFT-S1 + 100 step SFT-S2 + 1 bench × 30q eval，**走通整条 pipeline 拿到一个真实分数**，再放大。
2. 一次最长投入 ≤ 12h 没 mid-eval 反馈则必须停下评估。

### 6.2 Paper alignment audit 来太晚

**问题**：训了几十小时才系统对照 paper，发现 backbone / LoRA r / chunk_size / 训练范式都偏。

**防止**：
1. **Day-1 必产出 `PAPER_ALIGNMENT_AUDIT.md`**：列 paper 所有 hyperparam 跟我们的对比表，所有偏离记录"为什么 + 风险"。
2. 每次开新分支 / 改大 hyperparam 前，更新这个 audit 一次。

### 6.3 没 reviewer

**问题**：一个人闷头跑，几个明显错误（r=16、单 backbone、curriculum 跳跃）其实 review 一眼能看出。

**防止**：
1. 任何 ≥ 24h 的训练，启动前必须有 1 个人 review launch 配置 + audit 文档。
2. ≥ 7 天的训练（如完整重训 instruct chain）必须有 design doc + 2 人 review。

---

## 7. Eval / 对比（P2）

### 7.1 Hybrid-oracle 来得太晚

**问题**：训完才想到测 LM 上限（hybrid-oracle 绕过 router 喂 gold docs），发现 LM 也弱。如果训中就跑，能更早发现 router 不背锅。

**防止**：
1. mini-eval 同时跑 2 模式：
   - (a) **真 router**：测 end-to-end pipeline
   - (b) **hybrid-oracle**：测 LM 上限
2. 两者差距大说明 router 不行；两者都低说明 LM 不行；两者都高才能上 final eval。

### 7.2 Vanilla baseline 没 Day-1 测出来

**问题**：vanilla 9B-Instruct 三 mode（oracle/BM25/no-ctx）的 baseline 是训完很久后才补测的。早测出来的话，"训完分数 < vanilla no-ctx" 这种红线能在 mid-eval 阶段就触发 kill。

**防止**：
1. **Day-0 必须先跑 vanilla 全 baseline**（约 30 min compute time）。
2. 写进 mid-eval 红线：`mid_eval_score < vanilla_no_ctx_avg × 0.93` → kill。

---

## 8. 下一轮训练 Checklist（必须勾完才能 launch）

### Day 0（启动前 24h 内必须完成）

- [ ] **Paper alignment audit 文档更新**：所有 hyperparam 对齐 paper，偏离项有书面理由
- [ ] **Vanilla baseline eval**：跑 9 bench × 100q × 3 mode（oracle/BM25/no-ctx），记录 AVG → 用作 mid-eval 红线
- [ ] **Pipeline smoke**：1 bench × 100 step CPT + 100 step SFT-S1 + 100 step SFT-S2 + 30q eval，**实际拿到一个 LLM-judge 数字**
- [ ] **数据 split audit**：跑 `audit_data_split.py`，确认 SFT train ∩ eval = ∅
- [ ] **Wandb project 建好**：所有 metric 配齐，alert webhook 到 Slack
- [ ] **Tmux session 建好**：`tmux new -s msa_cpt`，所有训练命令在 session 里跑
- [ ] **Launcher commit 到 git**：`scripts/launch_<exp>.sh` 加进 repo
- [ ] **Reviewer 签字**：launch 配置 + audit 文档至少有 1 人 review

### 训练中（每个 stage 持续监控）

每 200 step 自动跑：
- [ ] router_precision、router_recall（< 0.05 alarm，< 0.03 kill）
- [ ] empty_answer_rate（> 0.30 alarm，> 0.50 kill）
- [ ] loss rolling-1k-step std（< 0.05 + 持续 2k step → plateau alarm）

每 1000 step 自动跑：
- [ ] mini-eval：1 bench × 30q LLM-judge（< vanilla_no_ctx × 0.93 → kill）
- [ ] hybrid-oracle mini-eval：1 bench × 30q（看 LM 上限）

每 5000 step 自动跑：
- [ ] full mini-eval：3 bench × 30q（看跨 bench 一致性）
- [ ] NIAH long-context probe（看 MSA-attention 是否在学）

### 训练后（每个 ckpt promote 前）

- [ ] full 9 bench × 50q LLM-judge（hybrid-oracle 模式）→ 看 LM 上限
- [ ] full 9 bench × 50q LLM-judge（真 router 模式）→ 看 end-to-end
- [ ] vs 之前 ckpt 对比表（每个 bench 各 5 个 metric）
- [ ] 至少在 1 个新 bench / split 上验证（防 overfit eval set）
- [ ] **任一 bench 退化 > 10% → 不 promote，回退**

### 红线（任何一条触发立即 kill 训练）

| 红线 | 阈值 | 说明 |
| --- | --- | --- |
| router_precision | < 0.03（3 次 mid-eval 持续）| router 没在学 |
| empty_answer_rate | > 0.50 | LM 没学会三段式 |
| loss plateau | rolling 2k-step std < 0.05 + 均值变化 < 0.01 | 已收敛或学崩 |
| mini-eval LLM-judge | < 1.98（vanilla no-ctx × 0.93）| 训练在退化 |
| GPU OOM 重启 | > 2 次 / 24h | 配置炸了 |

---

## 9. 一句话总结

> **过去这几个月失败的根本不是技术不行，是"没在训中知道自己崩了"**。底层模型架构 / hyperparam 哪怕全错，只要训中有 LLM-judge 红线，最多浪费 1000 step（1 小时）就 kill 重来；但因为没监控，**全错的配置跑了 65 小时**才发现。下一轮训练**先建监控基础设施，再 kick off 训练**，比任何架构 / 数据优化都更值得投入。
