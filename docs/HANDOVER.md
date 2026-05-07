# MSA × Qwen3.5-9B 项目交接文档（William 接手）

> 交接日期：2026-04-24
> 上一接手人：Zeyu
> 项目：复现 EverMind/MSA paper（arXiv:2603.23516）的 9B 长上下文模型
> 仓库：本仓库 https://github.com/zeyuyuyu/msa-minimind （fork from william-0g/msa-minimind）
> 当前活跃分支：`docs/training-reports`

---

## 0. TL;DR（30 秒读完）

1. **跑过两条训练链都没达到 paper 水平**：
   - **base 链**：`Qwen3.5-9B-Base` → CPT → SFT-S1 → SFT-S2，全部完成。最终 LLM-judge **1.503/5**（远低于 paper MSA-4B-S2 的 2.21）。
   - **instruct 链**：`Qwen3.5-9B-Instruct` → CPT-5a (51k step) → CPT-5b (30k step, killed flat loss) → Fork-S1 (1k step smoke, 100% empty answer)。已暂停。
2. **根因已定位（见 §6）**：router 训练严重不充分（precision ≈ 0.03 ≈ random）+ LM 自身也偏弱。
3. **hybrid-oracle 9-bench 50q eval 已全部完成**（2026-05-07 07:17 UTC）：9-bench AVG = **2.4222**，超过 vanilla no-ctx 2.12（+0.30）但远低于 vanilla oracle 3.90（-1.48）。当前 cvm-rl 上没有正在跑的 job。
4. **下一步推荐方案（见 §7）**：先做 router-only finetune（48h, sanity check）→ 不行就完整重训 instruct chain（约 7-10 天）。
5. **重要文档（必读，按重要性排序）**：

   | 文档 | 用途 |
   | --- | --- |
   | [`docs/LESSONS_LEARNED.md`](./LESSONS_LEARNED.md) | ⭐ **最重要**：7 类失败原因 + 防止再犯 checklist + 红线规则 |
   | [`docs/PAPER_ALIGNMENT_AUDIT.md`](./PAPER_ALIGNMENT_AUDIT.md) | paper / 官方 repo / 我们 implementation 三方对比 + 重训 blueprint |
   | [`docs/APPLE_TO_APPLE_EVAL_REPORT.md`](./APPLE_TO_APPLE_EVAL_REPORT.md) | 4 组对照实验结果 + 根因数据支撑 |
   | [`docs/BASE_TRAINING_REPORT.md`](./BASE_TRAINING_REPORT.md) | base 链全流程（含失败原因 5 条） |
   | [`docs/INSTRUCT_TRAINING_REPORT.md`](./INSTRUCT_TRAINING_REPORT.md) | instruct 链 CPT/Fork-SFT 细节 |

---

## 1. 项目目标

复现 paper 的 100M token long-context demo，使 MSA × Qwen3.5-9B 在 9 个长上下文 QA bench 上达到或接近 paper Table 2 的 MSA-9B-S2 数字（论文 4B 模型 AVG 约 2.21，9B 应高一些）。具体 bench：

```
musique  hotpotqa  nature_questions  msmarco_v1  2wikimultihopqa
hipporag_popqa  hipporag_narrative  dureader  triviaqa_06M
```

评分指标：LLM-judge 0-5（`google/gemini-2.5-flash` via OpenRouter，prompt 来自 paper 官方 `scripts/llm_judge_evermind.py`）。

---

## 2. 当前结果总览（最重要！）

### 2.1 训练前 baseline（vanilla `Qwen3.5-9B-Instruct`，无任何训练，9 bench × 100q）

完整数据来自 [`/workspace/eval_vanilla_instruct/full/*/llmscore_summary.json`（william-dev）](#5-机器--路径)：

| Mode | musique | hotpotqa | NQ | msmarco_v1 | 2wikimqa | popqa | narrative | dureader | triviaqa | **AVG** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **+ oracle** (gold docs in prompt) | 3.58 | 4.54 | 3.92 | 3.82 | 4.02 | 3.63 | 3.25 | 3.91 | 4.43 | **3.90** |
| **+ BM25 top-5** | 1.36 | 3.35 | 3.51 | 3.04 | 2.37 | 2.42 | 1.82 | 0.73 | 4.28 | **2.54** |
| **+ no-context** (parametric only) | 1.02 | 2.13 | 2.05 | 2.70 | 2.50 | 1.90 | 1.02 | 1.99 | 3.74 | **2.12** |

### 2.2 base 链 SFT-S2 训完后（9 bench × 50q，原始 evaluation 流程）

| Bench | empty_rate | router_precision | LLM-judge |
| --- | --- | --- | --- |
| musique | 0.91 | 0.028 | 0.46 |
| 9-bench AVG | 0.40+ | 0.04 | **1.503** |

详见 [BASE_TRAINING_REPORT.md §5](./BASE_TRAINING_REPORT.md)。

### 2.3 base 链 SFT-S2 + hybrid-oracle（绕过 router，强行喂 gold docs）

cvm-rl 9 bench × 50q 已全部完成（`/workspace/eval_hybrid_full/llmscore_summary.json`，2026-05-07 07:17 UTC）：

| Bench | LLM-judge | vs vanilla+oracle |
| --- | --- | --- |
| musique 50q | 1.54 | -2.04 |
| hotpotqa 50q | 3.48 | -1.06 |
| nature_questions 50q | 1.24 | **-2.68** ⚠ |
| msmarco_v1 50q | 2.88 | -0.94 |
| 2wikimultihopqa 50q | 2.20 | -1.82 |
| **hipporag_popqa 50q** | **4.46** | **+0.83** ✅ 唯一反超点 |
| **hipporag_narrative 50q** | **0.54** | **-2.71** ⚠⚠ 最差 |
| dureader 50q | 2.64 | -1.27 |
| triviaqa_06M 50q | 2.82 | -1.61 |
| **9-bench AVG** | **2.4222** | -1.48 |

详见 [APPLE_TO_APPLE_EVAL_REPORT.md §3.4](./APPLE_TO_APPLE_EVAL_REPORT.md)。

### 2.4 一句话结论

> **base 链训完产生绝对退化**（SFT-S2 真实 1.503 < vanilla no-ctx 2.12）。即使把 router 完全修好（hybrid-oracle 9-bench AVG 2.4222）也只能勉强超过 vanilla no-ctx（+0.30），距离 vanilla oracle (3.90) 还差 1.48 分。**LM 本身也被训弱了**，narrative / NQ / 2wiki 三个 bench 在 oracle 下都掉超过 1.8 分。

---

## 3. 训练链概览

### 3.1 Base 链（已结束，LoRA r=16, alpha=32）

| Stage | Steps | Tokens | 时间 | 数据集 | Ckpt |
| --- | --- | --- | --- | --- | --- |
| CPT  | 7000  | 35.84M | 15h | MS MARCO v2.1 (50K facts) | `/workspace/runs/cpt_qwen3_5_*/qwen3_5_msa_cpt.pt` |
| SFT-S1 | 7000 | 35.84M | 8.5h | sft_mix（5 QA bench mix） | `/workspace/runs/sft_s1_qwen3_5_0425_1439/qwen3_5_msa_sft_s1.pt` |
| **SFT-S2** | 6000 | 212.58M | **42h** | sft_mix + curriculum 64→512 docs | `/workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt`（**这是当前 evaluation 用的 ckpt**） |

启动脚本：`scripts/long_cpt_qwen3_5.sh` / `sft_stage1_qwen3_5.sh` / `sft_stage2_qwen3_5.sh`

### 3.2 Instruct 链（已暂停，LoRA r=64, alpha=128）

| Stage | Steps | Tokens | 状态 | Ckpt |
| --- | --- | --- | --- | --- |
| CPT-5a | 51k / 120k 计划 | 437M | 中止于 51k（约 94h） | `/workspace/runs/p5_shard1/p5_step50k.pt` |
| CPT-5b | 30k / 60k 计划 | 259M | killed（flat loss > 16k step）+ 后期被 user 要求让出 H200 | `/workspace/runs/p5b_shard1/p5b_step20k.pt` |
| Fork-S1 smoke | 1000 | 5M | smoke 100% empty answer，未 promote | `/workspace/runs/sft_s1_qwen3_5_*` |
| Fork-S2 | 25 step | — | killed | — |

启动脚本：`/workspace/launch_p5_cpt.sh` / `launch_p5b_cpt.sh` / `launch_fork_s1.sh` / `launch_fork_s2.sh`

详见 [INSTRUCT_TRAINING_REPORT.md](./INSTRUCT_TRAINING_REPORT.md)。

---

## 4. 关键代码索引（容器内绝对路径，下同）

### 4.1 训练

| 文件 | 用途 |
| --- | --- |
| `/workspace/msa-minimind/msa/training/{cpt,sft_s1,sft_s2}.py` | 三阶段训练主循环 |
| `/workspace/msa-minimind/msa/model/qwen3_5_msa.py` | MSA-Qwen3.5 模型定义（LoRA + MSA-attention 注入） |
| `/workspace/msa-minimind/msa/data/sft_mix.py` | sft_mix 数据集组装（5 bench mix） |
| `/workspace/msa-minimind/scripts/long_cpt_qwen3_5.sh` | base CPT launcher |
| `/workspace/msa-minimind/scripts/sft_stage{1,2}_qwen3_5.sh` | base SFT launcher |
| `/workspace/launch_p5*.sh` | instruct CPT launcher（不在 repo 里，在容器 /workspace 根） |

### 4.2 推理 / Evaluation

| 文件 | 用途 |
| --- | --- |
| `/workspace/msa-minimind/msa/inference/sparse_generator.py` | MSA 推理核心：通过 `pooled_cache` 注入 docs，单 forward 三段式生成 |
| `/workspace/msa-minimind/msa/inference/router_engine.py` | `EncodedCorpus`（K/V/KR 缓存）+ `gather_topk_docs` |
| `/workspace/msa-minimind/msa/inference/prompt_template.py` | `build_msa_train_prompt` + `parse_msa_train_response`（三段式 part_a/b/c） |
| `/workspace/msa-minimind/scripts/bench_evermind_aligned_v2.py` | 主 evaluation 脚本（用 router 真实 retrieve） |
| `/workspace/msa-minimind/scripts/bench_msa_oracle_hybrid.py` | **hybrid-oracle**：绕过 router，强行注入 gold docs 到 `pooled_cache`（diagnostic 用） |
| `/workspace/msa-minimind/scripts/bench_vanilla_instruct.py` | vanilla 9B-Instruct + RAG 三 mode（oracle/BM25/noctx） |
| `/workspace/msa-minimind/scripts/llm_judge_evermind.py` | LLM-judge 0-5 评分（OpenRouter gemini-2.5-flash） |

### 4.3 关键概念

- **三段式输出**：MSA 训练范式是 `single-pass`，模型在一次 forward 里输出
  - **part_a**: `[1] [3] [5]` — 选 doc IDs
  - **part_b**: `1. <doc1 全文> 2. <doc3 全文> 3. <doc5 全文>` — 复述选中 doc 内容
  - **part_c**: `<End-of-Retrieve> answer` — 最终答案
- **`pooled_cache`**：MSA-attention 把 docs 的 K/V 预编码后塞进 attention 的 sparse mask，不走 prompt token 流。所以 evaluation 不能像普通 RAG 那样把 docs 拼到 prompt。
- **paper 是 `multi-round interleave`**：模型每轮只输出 part_a，系统注入 part_b text，再输出 part_c。我们 implementation 用的是 William 的 single-pass。详见 [PAPER_ALIGNMENT_AUDIT.md §11](./PAPER_ALIGNMENT_AUDIT.md)。

---

## 5. 机器 & 路径

> 两台都是 **NVIDIA H200 80G HBM3e × 1**（不是 H100，原文档已修正）+ 4.9T 数据盘，driver 570.172.08，挂在 Phala dstack TEE 上，SSH 走 port 443 + openssl ProxyCommand。

### 5.1 cvm-rl —— 我们 MSA 训练 / hybrid-oracle eval 主战场

#### SSH 配置（贴到 William 自己 `~/.ssh/config`）

```
Host cvm-rl
    HostName 670238bd987a441f48c007ec424afe8a688d3fe4-22.dstack-pha-in2.phala.network
    Port 443
    User root
    IdentityFile ~/.ssh/<your_private_key>      # 对应 william.wu@0g.ai 公钥的私钥
    ProxyCommand openssl s_client -quiet -connect %h:%p 2>/dev/null
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

> William 的 `william.wu@0g.ai` 公钥已经 append 到 `cvm-rl:/root/.ssh/authorized_keys`（perm 600 已配，跟 zeyu 自己的 key 共存）。

#### 容器

```bash
docker exec -it msa-dev bash       # 唯一一个 dev 容器
```

#### 容器环境（cvm-rl `msa-dev`）

| 项 | 值 |
| --- | --- |
| Python | 3.11.11（`/opt/conda/bin/python`） |
| torch | 2.6.0+cu124 |
| transformers | 5.3.0 |
| flash_attn | 2.7.4.post1 |
| 已安装 | `accelerate`、`peft`、`trl`（看 `pip list`） |
| **未安装** | wandb / sglang / vllm（如要监控 loss 用本地 tensorboard 或自己加 wandb） |
| HF cache | `/root/.cache/huggingface/`（23G） |
| 磁盘 | 4.9T 总，已用 1.5T (29%) |

> **没有 tmux/screen session**！我们之前的 hybrid eval 是用 `docker exec ... &` 跑的（log 写到 `/workspace/eval_hybrid_full/run.log`）。如果训练，建议自己 `tmux new -s sft_s2` 再跑，避免 ssh 断了 job 死。

#### 容器内 `/workspace` 目录全图（按重要性）

```
/workspace/
  ├─ msa-minimind/                  ← 仓库（origin = william-0g/msa-minimind），注意 uid=1000 不是 root
  │   ├─ msa/                       ← model + training + inference 全部 python module
  │   └─ scripts/                   ← bench_msa_oracle_hybrid.py / bench_evermind_aligned_v2.py / 等
  ├─ evermind_msa/                  ← 🔑 paper 官方 repo (https://github.com/...)，已 clone，inference 部分代码
  ├─ evermind_venv/                 ← paper 官方 repo 的独立 venv（如果想跑 paper inference 用这个）
  │
  ├─ qwen35_base/                   ← Qwen3.5-9B-Base HF weights
  ├─ qwen35_instruct/               ← Qwen3.5-9B-Instruct HF weights
  │
  ├─ runs/                          ← 所有训练 ckpt + log
  │   ├─ sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt    ← base 链 final（358MB LoRA only）⭐ 当前 eval 用这个
  │   ├─ sft_s1_qwen3_5_0425_1439/qwen3_5_msa_sft_s1.pt    ← base 链 SFT-S1 中间产物
  │   ├─ p5_shard1/                                         ← instruct CPT-5a（43GB total，含多个 step 快照）
  │   ├─ p5b_shard1/                                         ← instruct CPT-5b（13GB）
  │   └─ sft_s3_*/、sft_s2_smoke_*/、auto_*.log              ← 一些早期 smoke / 失败重跑（可忽略）
  │
  ├─ encoded_corpora/               ← 🔑 380G！所有 bench 的预编码 K/V/KR cache
  │   │   （chunk=32, top_k=8。如果改 chunk/k 必须全部重 encode）
  │   ├─ musique_99999/             ← 9 个评测 bench 各一份 (_99999 = 全 corpus，不限 size)
  │   ├─ hotpotqa_99999/、nature_questions_99999/、msmarco_v1_99999/
  │   ├─ 2wikimultihopqa_full/、hipporag_popqa_full/、hipporag_narrative_full/
  │   ├─ dureader_full/、triviaqa_06M_99999/
  │   ├─ ms_100M_50k/、ms_100M_full/、ms_50M_int4/...   ← MS MARCO 多份（CPT 训练数据）
  │   └─ pg_32K/, pg_64K/, pg_128K/, pg_256K/, pg_512K/, pg_1000K/
  │                                  ← 🔑 RULER/NIAH haystack（paper Fig 4 用的，长度 32K-1M）
  │
  ├─ msa_bench/                     ← 9 bench raw json 数据（query + gold doc 标注）
  ├─ cpt_shards_paper/shard_1/      ← paper-aligned CPT shard（KaLM + ST mix），3.1G
  ├─ shard_1_paper.tar              ← 同上压缩包，5.4G
  ├─ shard.tgz                      ← 旧 base 链 CPT shard，3.6G
  ├─ haystacks/                     ← NIAH 原始文本
  │
  ├─ eval_hybrid_full/              ← ⭐ 当前 in-flight eval（hybrid-oracle 9 bench × 50q）
  ├─ eval_evermind_aligned_v2/      ← base SFT-S2 真实 router eval（1.503 那次）
  ├─ eval_vanilla_base/             ← vanilla Qwen3.5-9B-Base eval
  │
  ├─ launch_p5_cpt.sh / launch_p5b_cpt.sh / launch_sft_s1.sh   ← instruct 链各阶段 launcher（不在 git！）
  ├─ run_bench_on_ckpt.sh / run_llm_judge.sh / auto_eval_*.sh   ← 自动化脚本
  └─ minimind/、msa-qwen/                                        ← 旧版本，可忽略
```

#### 一些进程层面的状态

- 当前长期跑的进程只有 hybrid-oracle eval（在 `eval_hybrid_full/run.log` 后台），没有训练 job。
- 有不少 `[python] <defunct>` 僵尸进程（早期 OOM / kill 留下，不影响 GPU）。

---

### 5.2 william-dev —— vanilla baseline / fork-SFT 实验机

#### SSH

```bash
ssh william-dev    # William 自己的 SSH 配置（HostName ≈ dc85116b... .dstack-pha-in2.phala.network:443）
docker exec -it compassionate_austin bash    # 容器名是这个，不是 msa-dev
```

#### 容器环境（william-dev `compassionate_austin`）

| 项 | 值 |
| --- | --- |
| Python | 3.12.3（系统 `/usr/bin/python`，**不是 conda**） |
| torch | 2.9.1（比 cvm-rl 新一档）|
| transformers | 5.3.0 |
| flash_attn | 4.0.0b5（`flash-attn-4`）|
| 已装 | `sglang 0.5.10rc0`、`peft 0.18.1`、`accelerate 1.6.0`、`wandb 0.26.0`、`trl 0.24.0`、`flashinfer-python 0.6.6` |
| HF cache | `/root/.cache/huggingface/`（**177G**，下过很多模型，有 sglang 拉的全套）|
| 磁盘 | 4.9T 总，已用 2.3T (46%) |

> **环境差异警告**：cvm-rl 用 conda + torch 2.6，william-dev 用系统 py3.12 + torch 2.9。**MSA 训练 / eval 我们一直在 cvm-rl 跑，因为 transformers 5.3 + torch 2.6 是验证过 work 的组合**。william-dev 主要用来跑 vanilla baseline + fork-SFT 实验。如果在 william-dev 跑训练前最好先小 smoke 测试。

#### william-dev tmux 状态（重要！）

```
0  detached   ← William 自己的 swe-bench / sglang 工作
1  detached   ← William sglang_covenant
2  ATTACHED   ← William 当前在用
8  detached
12 detached
13 detached
```

> **6 个 tmux 都是 William 自己的工作**（sglang server、swe-bench eval、Covenant-72B 等），**不是我们的 MSA 训练**。我们 fork-SFT 实验是用 `docker exec ... nohup &` 跑的，已经全部 kill 干净。**William 接手时不要误删自己的 tmux**。

#### 容器内 `/workspace` 目录

```
/workspace/
  ├─ msa-minimind/                              ← 跟 cvm-rl 同步过的代码
  ├─ qwen35_instruct/                           ← Qwen3.5-9B-Instruct（vanilla 用）
  ├─ encoded_corpora/                           ← 部分 bench（够跑 vanilla eval 即可）
  ├─ msa_bench/                                 ← 同 cvm-rl
  │
  ├─ runs/                                      ← fork-SFT 试验产物（smoke 失败，可删）
  │   ├─ sft_s1_qwen3_5_*/                      ← Fork-S1 1k step
  │   └─ sft_s2_*/                              ← Fork-S2 25 step（killed）
  │
  ├─ eval_vanilla_instruct/                     ← ⭐ vanilla baseline 全部结果在这
  │   ├─ full/oracle/llmscore_summary.json     ← AVG 3.90
  │   ├─ full/bm25/llmscore_summary.json       ← AVG 2.54
  │   ├─ full/noctx/llmscore_summary.json      ← AVG 2.12
  │   └─ bm25_smoke/                            ← 早期 30q smoke
  │
  ├─ launch_fork_s1.sh / launch_fork_s2.sh     ← 已结束的 fork SFT launcher
  ├─ auto_eval_daemon.sh / manual_eval.sh      ← 自动化 eval 脚本
  └─ swe / swe_qwen2_unres / outputs/          ← William 自己的 swe-bench 工作（不要动）
```

---

### 5.3 我本地工作区（zeyu，仅用于编辑 docs / 推 git）

```
/home/zeyu/msa-qwen/msa-minimind/    # git working tree
  ├─ docs/                            # 5 个 markdown 文档都在这里
  │   ├─ HANDOVER.md                  ← 你正在看的
  │   ├─ PAPER_ALIGNMENT_AUDIT.md
  │   ├─ APPLE_TO_APPLE_EVAL_REPORT.md
  │   ├─ BASE_TRAINING_REPORT.md
  │   └─ INSTRUCT_TRAINING_REPORT.md
  ├─ msa/、scripts/                   # 跟 cvm-rl 容器内 msa-minimind/ 同步
  └─ ...
```

#### Git remote

```
origin   = https://github.com/william-0g/msa-minimind.git    # 上游
myfork   = https://github.com/zeyuyuyu/msa-minimind.git      # 我 push 到这里
当前 branch = docs/training-reports                          # 所有 docs 改动都在这
```

> 已开 PR：`docs: training reports` 系列。William 接手后可以选择 merge 到自己的 main 或者直接 cherry-pick 关心的 commit。

---

### 5.4 速查：跨机器 / 跨容器路径对照

| 内容 | cvm-rl `msa-dev` | william-dev `compassionate_austin` | 我本地 |
| --- | --- | --- | --- |
| 仓库代码 | `/workspace/msa-minimind/` | `/workspace/msa-minimind/` | `/home/zeyu/msa-qwen/msa-minimind/` |
| Qwen3.5-9B-Base | `/workspace/qwen35_base/` | ❌ 没下 | ❌ |
| Qwen3.5-9B-Instruct | `/workspace/qwen35_instruct/` | `/workspace/qwen35_instruct/` | ❌ |
| 训练 ckpt | `/workspace/runs/` | `/workspace/runs/`（仅 fork-SFT smoke） | ❌ |
| EncodedCorpus | `/workspace/encoded_corpora/`（380G,全） | `/workspace/encoded_corpora/`（部分） | ❌ |
| Bench raw data | `/workspace/msa_bench/` | `/workspace/msa_bench/` | ❌ |
| paper 官方 code | `/workspace/evermind_msa/` | ❌ | `/tmp/evermind_msa/` |
| eval 输出 | `/workspace/eval_*/` | `/workspace/eval_vanilla_instruct/` | ❌ |

---

## 6. 根因分析（精简版）

> 完整推理链见 [PAPER_ALIGNMENT_AUDIT.md](./PAPER_ALIGNMENT_AUDIT.md) 和 [APPLE_TO_APPLE_EVAL_REPORT.md](./APPLE_TO_APPLE_EVAL_REPORT.md)。

### P0 — Router 训练严重不充分

base SFT-S2 ckpt 在 musique 上 router precision = **0.028**（gold ratio 0.04 → 接近 random）。SFT-S2 课程从 64→512 docs 增加得太快、loss 信号被 part_b 吃掉，router 没学到检索。

**证据**：hybrid-oracle 把 router 跳过、直接喂 gold docs，9-bench AVG 从 1.503 → 2.4222（+0.92）。

### P1 — LM 自身也偏弱

hybrid-oracle 9-bench AVG = 2.4222 < vanilla oracle 3.90，差 1.48 分。即使检索完美，LM 本身的回答能力也明显输给训练前的 9B-Instruct。后跑的 hipporag_narrative (0.54) 把均值显著拉下（-2.71 vs vanilla oracle），说明长文叙事 QA 是 SFT 退化最严重的场景。

**怀疑成因**（按重要性）：
1. **LoRA rank 太小**（r=16 vs paper r=64）→ adaptation capacity 不够。
2. **Backbone 选错**（Base vs paper Instruct-2507）→ chat / instruction-following 先验缺失。
3. **CPT 数据量太少**（35.84M vs paper 数百 B token）→ MSA-attention 没充分预训练。
4. **三段式 prompt 在 short-answer 上有副作用**（NQ/msmarco_v1 这种 1-2 词答案被 part_b 1300+ char 复述污染）。

### P2 — 训练范式 single-pass vs paper multi-round interleave

William 实现是 single-pass（part_a/b/c 一次出），paper / 官方 inference 是 multi-round interleave（每轮只生成 doc-id，系统注入 doc text，再生成 answer）。原以为这是 P0，但 hybrid-oracle 数据证明 single-pass 不是 fatal，只是工程效率上不如 multi-round（part_b 占 max_new_tokens budget）。

### ❌ 不是问题的：bench 数据没进训练 / oracle 注入实现 bug

都验证过了，排除。

---

## 7. 推荐下一步路线

### Phase A（优先）— Router-only Finetune（48h, sanity check）

**假设**：LM 已经能用（hybrid-oracle 9-bench AVG 2.4222 已超 vanilla no-ctx 2.12），只要把 router 修到 paper 水平，理论天花板就是这个 2.42。但因为真 router precision = 0.028 ≈ random，Phase A 即使把 router 修到完美也只能拿到 ~2.4，绝对达不到 paper MSA-9B 等效估计的 ~3.0。

**做法**：
1. 冻结 base SFT-S2 ckpt 的 LM LoRA 参数。
2. 只 finetune router 子模块（在 `msa/model/qwen3_5_msa.py` 里 router projection layers）。
3. 用 sft_mix + 强制 router teacher forcing（让 router precision 直接对齐 gold doc indices，而不是只靠 LM 端 loss）。
4. 跑 1k-3k step，每 500 step eval router precision/recall。
5. 目标：router precision > 0.5（musique 4 gold / 8 retrieved 至少对 2 个）。

**判定**：
- router precision > 0.5 后跑全 9-bench eval → AVG > 2.5 → Phase A 成功，写 paper。
- router precision 上去了但 AVG 没起来 → LM 真不行 → 走 Phase B。

### Phase B（兜底）— 完整重训 instruct chain（7-10 天）

按 [PAPER_ALIGNMENT_AUDIT.md §12 重训 blueprint](./PAPER_ALIGNMENT_AUDIT.md) 的配置：

| 项 | 取值 | 理由 |
| --- | --- | --- |
| Backbone | `Qwen3.5-9B-Instruct` (or 2507) | paper 官方选择 |
| LoRA | r=64, alpha=128 | 跟 paper 一致 |
| MSA chunk | 64 | paper 用 64 |
| top-k | 16 | paper 用 16 |
| CPT 数据 | KaLM + ST mix（已 shard 在 `/workspace/encoded_corpora/`） | paper-aligned |
| CPT step | 至少 80k | 之前 51k 还在降，提前停早了 |
| SFT-S1 step | ≥ 5k | 之前 1k smoke 不够 |
| SFT-S2 curriculum | 64→128→256→512 平滑 | 别一次跳 |

如果做这条路线，**强烈建议同时实现 paper 的 multi-round interleave inference**（对比 single-pass 看哪个 bench 提分多），代码模板在 `/tmp/evermind_msa/src/msa/generate.py`（官方 repo clone）。

### Phase C（可选）— 直接 fork paper 4B ckpt

paper 在 HuggingFace 上有公开 ckpt（搜 `evermind-msa-4b`）。直接 load + 在我们的 9 bench eval 上跑，作为**真实 paper baseline**。这能搞清楚"我们 implementation 是否 buggy" vs "我们 training 没复现到位"。如果 paper ckpt 在我们的 eval 上能拿 2.21 → implementation 对，我们训得不行；如果 paper ckpt 在我们 eval 上也低 → implementation 有 bug。

---

## 8. 操作 Cheat Sheet

### 8.1 看 hybrid-oracle eval 进度（cvm-rl）

```bash
ssh cvm-rl 'docker exec msa-dev bash -lc "ls /workspace/eval_hybrid_full/*.json | grep -v llmscore | grep -v summary; tail -3 /workspace/eval_hybrid_full/run.log"'
```

### 8.2 触发一次 LLM-judge

```bash
ssh cvm-rl 'docker exec -e OPENROUTER_API_KEY=<your_key> msa-dev bash -lc \
  "cd /workspace/msa-minimind && python scripts/llm_judge_evermind.py \
   --in_dir /workspace/eval_hybrid_full \
   --benches musique,hotpotqa,nature_questions,msmarco_v1,2wikimultihopqa,hipporag_popqa \
   --backend openrouter --model google/gemini-2.5-flash"'
```

### 8.3 跑 hybrid-oracle 单 bench（示例）

```bash
docker exec msa-dev bash -lc "cd /workspace/msa-minimind && python scripts/bench_msa_oracle_hybrid.py \
  --qwen3_5_path /workspace/qwen35_base \
  --resume_ckpt /workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt \
  --encoded_root /workspace/encoded_corpora \
  --benches musique \
  --num_queries 50 \
  --out_dir /workspace/eval_hybrid_full"
```

### 8.4 跑 vanilla baseline（william-dev）

```bash
docker exec compassionate_austin bash -lc "cd /workspace && python bench_vanilla_instruct.py \
  --model_path /workspace/qwen35_instruct \
  --bench musique \
  --mode oracle \
  --num_queries 100 \
  --out_dir /workspace/eval_vanilla_instruct/full/oracle"
# mode 可选: oracle | bm25 | noctx
# bench 一次跑一个，不能逗号分隔（脚本限制）
```

### 8.5 启动训练（参考 base SFT-S2）

```bash
docker exec msa-dev bash -lc "cd /workspace/msa-minimind && bash scripts/sft_stage2_qwen3_5.sh"
# 会写到 /workspace/runs/sft_s2_qwen3_5_<date>/
# 监控: tail -f /workspace/runs/sft_s2_qwen3_5_<date>/train.log
```

### 8.6 OpenRouter API key

不在 repo 公开。需要时用自己的 key（OpenRouter 上注册免费即得 $1 credit，足够把 9 bench × 50q LLM-judge 跑十几轮）。

成本参考：9 bench × 50q × `gemini-2.5-flash` judge ≈ $0.30 / 次。整套 retraining + 多轮 eval 合计预算 < $10。

调用方式：`OPENROUTER_API_KEY=<your_key>` 透过 `docker exec -e OPENROUTER_API_KEY=...` 注入容器即可，参考 §8.2。

---

## 9. 红线监控（必须看的指标）

训练时每 200 step 跑一次 mini-eval（10 query × 1 bench），任一红线触发立刻 kill：

| 指标 | 阈值 | 触发后果 |
| --- | --- | --- |
| `router_precision` | < 0.05 | router 没在学 → kill |
| `empty_answer_rate` | > 0.30 | LM 没学会三段式 → kill |
| `loss` 平台 | 连续 > 2k step 不降 | 收敛了 / 学崩了 → kill |
| `LLM-judge`（每 1000 step 跑 1 bench × 30q） | < 1.98（vanilla no-ctx baseline 的 0.93x） | 训练在退化 → kill |

base 链就是因为没有第 4 项导致一直跑到 SFT-S2 训完才发现退化。

---

## 10. 已知坑 & 教训

1. **MSA 三段式 prompt 不能用普通 RAG eval 脚本**：必须用 `bench_evermind_aligned_v2.py` 或 `bench_msa_oracle_hybrid.py`，里面 prompt = "naked question"，docs 通过 `pooled_cache` 注入。
2. **Base backbone 缺 chat 先验** → instruction-following 全靠 SFT 学。如果训练数据里 instruction 比例小，模型不会按格式输出。后续选 Instruct 起步。
3. **LoRA r=16 太小**：base 链用了 r=16，instruct 链改成 r=64（paper 一致），后者 loss 曲线明显更陡。
4. **CPT 数据量必须管够**：35.84M token 远不够让 MSA-attention 收敛。下一轮至少 200M+。
5. **SFT-S2 curriculum 不能跳太快**：从 64 直接跳到 512 docs 是错的。下次 64→128→256→512 平滑过渡，每档 ≥ 1500 step。
6. **shell quoting 坑**：`docker exec` 嵌 bash -lc 嵌 python 命令时，长 string / SSH key 容易炸。统一改成"先写 shell 脚本，scp 到机器，再 docker exec 执行脚本"模式。
7. **Phala TEE 容器重启会换 host key** → `~/.ssh/config` 里必须设 `StrictHostKeyChecking no` + `UserKnownHostsFile /dev/null`。
8. **某些 bench 数据集需要专门处理**：
   - `nature_questions` 是 1-2 词短答 → 三段式 part_b 复述会污染 part_c 答案抽取。
   - `dureader` 是中文 → tokenizer 处理需要额外注意。
   - `triviaqa_06M` 文档数量大 → 编码 corpus 时间长（~30min/bench）。
9. **`encoded_corpora/` 必须重新生成才能换 chunk size**：现存的是 chunk=32, top_k=8。如果改 chunk=64 / top_k=16 要全部重 encode。
10. **OpenRouter API key 偶尔失效**：401 user not found 的时候不是 key 错，是 OpenRouter 那边 user 状态炸了，换 key 即可。

---

## 11. 当前 in-flight 任务

| Task | 位置 | 状态 | ETA |
| --- | --- | --- | --- |
| hybrid-oracle 9 bench × 50q | cvm-rl `/workspace/eval_hybrid_full/` | ✅ 完成 2026-05-07 07:17 UTC（AVG 2.4222） | — |
| ~~写 v4 报告（hybrid 9/9 全完整数据 + Phase A 推导）~~ | `APPLE_TO_APPLE_EVAL_REPORT.md` | ✅ 已更新到 v4（2026-05-07） | — |
| Phase A: router-only finetune | 未启动 | pending | ~48h（如果决定走，天花板 ~2.4） |
| Phase B: 完整 instruct chain 重训 | 未启动 | pending | ~7-10 天 |

---

## 12. 待 William 决策

1. **走 Phase A 还是直接 Phase B？**
   - Phase A 便宜但赌 LM 已经够用 → hybrid 9/9 完整数据显示 LM 在完美检索下也只 2.42，**Phase A 天花板就是 2.42**（绝对值），不可能更高。
   - Phase B 慢但结果可预期（LoRA r=64 + Instruct backbone + 充足 CPT 已被证明过 paper 走通）。
   - **我个人建议**：直接 Phase B。Phase A 投入 2 天天花板 2.4，远低于 paper 2.21（4B）的 9B 等效估计 ~3.0；而且 narrative (0.54) / NQ (1.24) 这种 oracle 下都崩的 bench，再修 router 也救不回来。
2. **要不要做 Phase C（fork paper ckpt + 在我们 eval 上跑）？** 这能彻底分清是 implementation bug 还是 training 没到位，~1 天即可完成。**强烈建议跑这一轮**作为决策依据。
3. **是否接受 single-pass 训练范式？** 如果决心追平 paper，应该把 multi-round interleave 也实现一份，对比两者哪个上限高。
4. **要不要切到 instruct-2507 backbone？** paper 用的是 2507 的 base，比 9B-Instruct 老一些但官方就是这个。

---

## 13. 联系 / Credentials

- 仓库 PR：所有 docs 改动都在 `docs/training-reports` branch（fork = `zeyuyuyu/msa-minimind`），已有 PR 上 `william-0g/msa-minimind`。
- cvm-rl SSH key：William 的 `william.wu@0g.ai` 公钥已在 `cvm-rl:~/.ssh/authorized_keys`（root 用户），perm 600 已配。
- william-dev：William 自有 access。
- OpenRouter：见 §8.6（个人 key，请在自己结算后换成 William 自己的）。
- HuggingFace：Qwen3.5 weights 已 download 到两台机器；不需要再下载。

---

## 14. 文件地图（Quick Reference）

```
本仓库 docs/
  ├─ HANDOVER.md                        ← 本文档（先读）
  ├─ LESSONS_LEARNED.md                 ← ⭐ 失败复盘 + 防止再犯 checklist
  ├─ PAPER_ALIGNMENT_AUDIT.md          ← paper vs ours 对比 + 重训 blueprint
  ├─ APPLE_TO_APPLE_EVAL_REPORT.md     ← 4 组 controlled experiments + 根因数据
  ├─ BASE_TRAINING_REPORT.md            ← base 链全流程
  └─ INSTRUCT_TRAINING_REPORT.md        ← instruct 链全流程

cvm-rl 容器关键 ckpt:
  /workspace/runs/sft_s2_qwen3_5_0426_0349/qwen3_5_msa_sft_s2.pt   ← base 链 final
  /workspace/runs/p5_shard1/p5_step50k.pt                          ← instruct CPT-5a
  /workspace/runs/p5b_shard1/p5b_step20k.pt                        ← instruct CPT-5b

eval 输出（按重要性）:
  /workspace/eval_hybrid_full/                       ← 当前 in-flight，最重要
  /workspace/eval_evermind_aligned_v2/               ← base SFT-S2 真实 router eval (1.503)
  william-dev:/workspace/eval_vanilla_instruct/full/ ← vanilla 3 mode baseline
```

---

> **如果只看一个文档（启动重训前必读）**：[`LESSONS_LEARNED.md`](./LESSONS_LEARNED.md)（含 §8 启动 checklist + 红线规则）。
> **如果想看重训 blueprint**：[`PAPER_ALIGNMENT_AUDIT.md` §12](./PAPER_ALIGNMENT_AUDIT.md)。
> **如果只看一组数据**：[`APPLE_TO_APPLE_EVAL_REPORT.md` §3](./APPLE_TO_APPLE_EVAL_REPORT.md)（4 组对照表）。
> **如果想直接复现 hybrid-oracle eval**：本文档 §8.3。
> **任何问题**：transcript 在 `agent-transcripts/f57e37db-fbb1-452c-849e-179d1b5aca1d`，含 100+ 轮迭代记录。
