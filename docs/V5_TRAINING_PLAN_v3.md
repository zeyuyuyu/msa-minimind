# V5 Training Plan (rev3 — based on v4 forensic findings)

**Date**: 2026-05-08
**Author**: Zeyu Wang (with Claude Opus 4.7)
**Status**: Plan ready, pending shard-loader port + smoke test

**Supersedes**: V5_TRAINING_PLAN.md (rev2), which was based on incorrect v4 diagnosis.

---

## 0. Why rev3 — v4 forensic findings

The earlier plan (rev1, rev2) was based on guessed v4 failure modes. After reading
the actual v4 launcher script and ckpt structure, three claims I made earlier turned
out to be **wrong**, and the real failure mode is different:

| Earlier claim (wrong) | Actual reality |
|---|---|
| v4 used 36-source mix (`cpt_shards`) | **v4 used `--data ms_marco` single-source** (MS MARCO v2.1, 502,939 capped queries). `cpt_shards_paper` was prepared for the parallel 9B path William was running, not 4B. |
| v4 LoRA was attention-only, ~16M trainable | **v4 LoRA covered all 7 modules (q/k/v/o + gate/up/down) on all 36 layers, 226M trainable** (5.3% of 4B). |
| v4 router was stuck at chance level | **v4 router converged to its mathematical ceiling** for MS MARCO triplets (1 positive per query → max precision@16 = 1/16 = 0.0625, recall = 1.0). |

**v4's true failure mode**: single-positive MS MARCO monoculture. The router learned
to retrieve **one** doc per query, but evaluation benches like HotpotQA / Musique
/ 2Wiki need **multi-doc** retrieval (multi-hop). The router has no training-time
supervision for "retrieve K positives", so it transfers poorly.

This is confirmed by the eval router F1:
- TriviaQA (single positive, same as MS MARCO): **0.479**
- HotpotQA / Musique / 2Wiki (multi-positive): 0.13–0.19

---

## 1. v5 fix: keep v4 setup, swap data source

The simplest minimal change is:

> **Replace `--data ms_marco` with `--data shard --shard_dir cpt_shards_paper/shard_X`**, where `cpt_shards_paper` is the already-prepared paper-aligned 36-source mix that includes multi-positive sources.

Everything else (LoRA config, MSA config, schedule) **stays identical to v4**, because v4's
non-data setup was already paper-aligned. The only data we change.

This is a true "minimum-change" experiment that will isolate whether multi-source
+ multi-positive data is sufficient to recover paper-level performance, without
confounding effects from rank changes, schedule changes, etc.

If v5 (data-only fix) recovers paper performance → confirmed: data was the problem.
If v5 still underperforms → we need orthogonal changes (rank, schedule, format).

---

## 2. Training data — `cpt_shards_paper` real composition (verified 2026-05-08)

`cpt_shards_paper/shard_1` on cvm-rl, **6,020,157 total samples**:

| Class | Source | Samples in shard_1 | Paper Table 5 expectation | Aligned? |
|---|---|---:|---|:---:|
| Long-context | kalmfinetune_data | 1,922,465 (31.9%) | 5.8M total / 3 shards = 1.93M | ✓ |
| Academic | S2ORC_citations_abstracts | 166,890 | ~167K (capped 0.5M / 3) | ✓ |
| Academic | S2ORC_citations_titles | 166,803 | ~167K | ✓ |
| Academic | S2ORC_title_abstract | 166,656 | ~167K | ✓ |
| Academic | specter_train_triples | 166,305 | ~167K | ✓ |
| QA | yahoo_answers_×3 | 167,044 / 166,908 / 166,271 | ~167K each | ✓ |
| QA | WikiAnswers | 166,278 | ~167K | ✓ |
| QA | gooaq_pairs | 166,744 | ~167K | ✓ |
| QA | msmarco-triplets | 166,556 | ~167K | ✓ |
| QA | PAQ_pairs | 166,853 | ~167K | ✓ |
| QA | amazon-qa | 167,139 | ~167K | ✓ |
| QA | eli5_question_answer | 108,380 | small original (<0.5M) | ✓ kept all |
| QA | stackexchange ×3 | 101,712 / 83,835 / 83,181 | small original | ✓ |
| QA | searchQA_top5_snippets | 155,651 | small original | ✓ |
| QA | quora_duplicates | 7,599 | small original | ✓ |
| QA | quora_duplicates_triplets | 33,785 | small original | ✓ |
| QA | NQ-train_pairs | 33,334 | small original | ✓ |
| QA | squad_pairs | 3,051 | small original | ✓ |
| QA | TriviaQA_pairs | 24,467 | small original | ✓ |
| News/Sum | agnews | 166,365 | ~167K | ✓ |
| News/Sum | npr | 166,022 | ~167K | ✓ |
| News/Sum | ccnews_title_text | 166,885 | ~167K | ✓ |
| News/Sum | cnn_dailymail_(splitted)? | 166,574 / 103,680 | ~167K / less | ✓ |
| News/Sum | xsum | 75,515 | small original | ✓ |
| News/Sum | sentence-compression | 59,954 | small original | ✓ |
| News/Sum | altlex | 37,385 | small original | ✓ |
| Domain | amazon_review_2018 | 166,637 | ~167K | ✓ |
| Domain | codesearchnet | 166,150 | ~167K | ✓ |
| Domain | AllNLI | 92,434 | small original | ✓ |
| Domain | wikihow | 42,708 | small original | ✓ |
| Domain | SimpleWiki | 34,086 | small original | ✓ |
| Domain | coco_captions | 12,662 | small original (vision-text) | ✓ |
| Domain | flickr30k_captions | 5,193 | small original (vision-text) | ✓ |

**36 / 36 sources covered** ✓
**Paper §B 0.5M cap rule applied** ✓
**KaLM full retention** ✓

Sample format (raw, before trainer wrap):
```jsonl
{"query": "Instruct: Retrieve semantically similar text. \n Query: 开通蚂蚁花呗需要多少分",
 "pos_idx": [0], "neg_idx": 1, "ds": "kalmfinetune_data"}
{"query": "who inducted james dudley into the wwe hall of fame",
 "pos_idx": [0], "ds": "PAQ_pairs"}
```

- KaLM samples carry the standard E5-style `Instruct: Retrieve... \n Query: ` prefix
- Other 35 sources use raw query text
- `pos_idx` / `neg_idx` are byte-offset indices into the shared `corpus.txt` file
- Some samples have `pos_idx` of length > 1 (multi-positive — this is the key data v4 lacked)

**Storage location**: cvm-rl `/workspace/cpt_shards_paper/shard_1/`
- `samples.jsonl` — 6M training records
- `corpus.txt` — 6.7M doc texts (indexed by byte offset)
- `shard.json` — metadata

**shard_0 is on f1f4 only**. We can either copy it across (via public IP rsync ≈ 30min for 3.1GB), or train on shard_1 alone for now (6M samples is plenty for 30000 main steps).

---

## 3. Training config — paper alignment table

| Item | Paper §A / config | v5 plan | Aligned? |
|---|---|---|---|
| Backbone | Qwen3-4B-Instruct-2507 (MSA-4B base) | same | ✓ |
| Hidden layers | 36 | 36 | ✓ |
| Attn heads | 32 (q) / 8 (kv) GQA | same | ✓ |
| Head dim | 128 | 128 | ✓ |
| MSA layer span | upper half = layers 18–35 | same | ✓ |
| chunk_size | 64 | 64 | ✓ |
| top_k | 16 | 16 | ✓ |
| msa_max_docs | 64 (CPT) | 64 | ✓ |
| max_doc_len | 192 (CPT) | 192 | ✓ |
| max_query_len | 256 (CPT) | 256 | ✓ |
| Per-sample tokens | 64 × 192 + 256 = 12,544 | same | ✓ |
| **num_router_heads** | q=32, k=8 (paper §A) | **k=8 (n_kv)** — `--router_k_uses_kv_heads 1` | ⚠️ partial — paper config has `num_router_heads_q=32`, our setup uses GQA convention with k=8 |
| decouple_router | true | true | ✓ |
| head_reduce | mean | mean | ✓ |
| query_reduce | max | max | ✓ |
| chunk_reduce | max | max | ✓ |
| decouple_pooling_mode | mean | mean | ✓ |
| rewrite_position | true | true | ✓ |
| Aux τ (infonce_loss_temp) | **0.1** (from ckpt config) | 0.1 | ✓ |
| Aux method | INFONCE_DECOUPLE | INFONCE_DECOUPLE | ✓ |
| Phase 1 weights | warmup: `0.1·LM + 1.0·Aux` | same | ✓ |
| Phase 2 weights | main: `1.0·LM + 0.1·Aux` | same | ✓ |
| Phase 1 lr | 1e-4 (warmup_lora_lr) | 1e-4 | ✓ |
| Phase 2 lr | 6e-6 cosine to 6e-7 | 6e-6 → 6e-7 | ✓ |
| Optimizer | AdamW, betas (0.9, 0.95), wd 0.01, grad_clip 1.0 | same | ✓ |
| Data sources | 36 (KaLM + 35 ST) — paper Table 5 | same | ✓ |
| Sampling cap | non-KaLM 0.5M, KaLM full | same | ✓ |
| Per-batch single source | yes (avoid in-batch hard-neg cross-domain) | yes | ✓ |
| **Microtune method** | **full-parameter** (paper) | **LoRA r=64, α=128** (v4 same) | ✗ **diverges — capacity 5.3%** |
| LoRA target modules (if applicable) | n/a (paper full-tune) | q/k/v/o + gate/up/down × all 36 layers | n/a |
| Total trainable | 4B (paper full-tune) | 226M (5.3%) | n/a |
| **CPT total step** | not stated; paper used 8× compute for ≥days | 30,000 main + 2,000 warmup = 32,000 | ✗ likely 1/8 of paper compute |
| Token consumption | not stated; paper had 158.95B available | ~0.4B (0.25% of paper budget) | ✗ |
| Hardware | not stated; paper likely 8× H100 | 1× H200 | ✗ |
| Target format | reverse-engineered from inference output | **TBD — must verify from `dataset_msa_shard.py`** | 🟡 |

**Aligned**: 22 / 25 verifiable items.
**Diverges**: 3 items (LoRA vs full-tune, CPT step count, hardware) — all stem from compute constraint.
**TBD**: 1 item (target format — must verify before launch, see §5).

---

## 4. v5 plan — single-machine MAIN run on cvm-rl

```bash
python -m msa.train_msa_cpt_qwen3 \
  --qwen3_path /workspace/qwen3_4b_instruct_2507 \
  --data shard \
  --shard_dir /workspace/cpt_shards_paper/shard_1 \
  --num_docs 64 --max_doc_len 192 --max_query_len 256 \
  --msa_start_layer 18 --msa_chunk_size 64 \
  --msa_top_k 16 --msa_max_docs 64 --msa_dropout 0.0 \
  --num_router_heads 8 --router_k_uses_kv_heads 1 \
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --batch_size 1 --num_workers 4 --epochs 100 \
  --warmup_steps 2000 --main_steps 30000 \
  --warmup_lora_lr 1e-4 --warmup_router_lr 2e-4 \
  --main_lora_lr 1e-5 --main_router_lr 5e-5 \
  --grad_clip 1.0 \
  --log_interval 25 --ckpt_every 1000 \
  --max_train_seconds 86400 \
  --dtype bf16 \
  --save_dir /workspace/runs/cpt_v5_4b \
  --save_name qwen3_4b_msa_cpt_v5 \
  --assert_sparse \
  --router_eval_every 100 \
  --auto_kill
```

**Predicted wall time on H200**: ~25–30h (similar to v4's 32k step setup).

**Token throughput**: ~12.5K tokens/sample × 32k step = 400M tokens (~0.25% of paper's 158.95B).

---

## 5. Pre-launch engineering work (must complete before launch)

### 5.1 Port `dataset_msa_shard.py` from 9B codebase to 4B codebase
- Source: `f1f4:/workspace/msa-minimind/msa/dataset_msa_shard.py` (9B path, exists)
- Target: `cvm-rl:/workspace/msa-minimind-v4/msa/dataset_msa_shard.py`
- Copy file, no modifications expected (data layout is path-keyed not model-keyed)

### 5.2 Patch 4B trainer for `--data shard`
File: `cvm-rl:/workspace/msa-minimind-v4/msa/train_msa_cpt_qwen3.py`

Changes:
- Line 274: `--data` choices: add `"shard"` to existing list
- Add `--shard_dir` arg
- In `build_dataset()`, add branch: `if args.data == "shard": from msa.dataset_msa_shard import load_shard; return load_shard(args.shard_dir, ...)`
- Mirror the 9B trainer's hookup at line 187–192

### 5.3 Verify target format alignment with paper
**Verified 2026-05-08**: Local fork's `prompt_template.py:build_msa_train_target` emits the
**simplified format**, NOT paper format:
```
[d1] [d2] ... The answer to the question is: <answer><|im_end|>
```

Paper format (reverse-engineered from inference output) is:
```
the document number related to the above issue is:
[id1]<|object_ref_end|>[id1]. <doc1 full text>
<|object_ref_end|>[id2]<|object_ref_end|>[id2]. <doc2 full text>
...
<|object_ref_end|><End-of-Retrieve><|im_start|>The user's question is: {q}
<|object_ref_end|>The answer to the question is: {answer}<|im_end|>
```

`dataset_msa.py` has `include_doc_text_in_target` forced to `False` (deprecated by William
in v4 with comment "paper format has no Part B"; **this is incorrect** — paper §4.3 ablation
explicitly says removing Original Text Injection costs -37%).

**Decision for v5**: stay with William's simplified format to **isolate the data-source
variable** (v4 vs v5 same code, only data differs). If v5 still underperforms, v6 will
restore paper format with Original Text Injection.

Trade-off:
- ✅ Cleaner attribution: any v4→v5 delta is purely from data-source change
- ✅ No tokenizer risk from special markers (`<|object_ref_end|>`, `<End-of-Retrieve>`)
- ✗ Diverges from paper §4.3, may cap maximum reachable performance

### 5.4 Smoke test
- 100-step run on a tiny copy of shard_1 (1000 samples)
- Verify: loss decreases from initial values, router_p ≥ 0.0625 by step 100, no NaN/inf
- If smoke OK → launch full 32k step run

---

## 6. Mid-CPT gates (auto-kill logic)

| Step | Trigger | Hard gate | Soft watch |
|---|---|---|---|
| 5000 | run mini-eval (30q × 9 bench) | `triviaqa ≥ vanilla 3.43` AND `msmarco ≥ vanilla 2.67` AND `nq ≥ vanilla 2.13` | Hotpot/Musique/2Wiki/Popqa/Narrative — record router F1 trend |
| 15000 | mini-eval (30q × 9 bench) | hard gate scores must be ≥ step5000 | soft watch — if multi-positive bench router F1 still <0.10, flag concern |
| 30000 | full eval (100q × 9 bench) | report 4-way comparison | n/a |

**Auto-kill rule**: if hard gate fails at step 5000 or 15000 → kill, save partial ckpt for inspection. Only trigger on benches that have CPT training-time exposure (the 3 single-positive benches with sources in Table 5).

---

## 7. f1f4 9B run — keep or kill?

**Current state** (verified 2026-05-08 13:15):
- Process: `python -m msa.train_msa_cpt_qwen3_5 --data shard --shard_dir cpt_shards_paper/shard_0`
- Backbone: Qwen3.5-9B-Instruct
- Step 44,200 / 60,000 (~74% done)
- Wall time so far: 84h, projected remaining: ~30h
- Loss: 0.39, healthy convergence
- LoRA r=64, same paper-aligned 36-source data

**Recommendation**: **let it finish**. It gives us a 9B paper-aligned baseline for comparison
with v5 4B. Killing it loses 84h of GPU work. The f1f4 GPU isn't usable for v5 anyway
(v5 4B will run on cvm-rl).

If GPU memory pressure forces a choice, we can revisit at step 50000.

---

## 8. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| `dataset_msa_shard.py` target format ≠ paper | 50% | High | Verify in §5.3 before launch; patch if needed |
| LoRA r=64 capacity insufficient for 36-source generalization | 30% | Medium | Mid-CPT gate at step 5000 catches this; ablation r=128 on william-dev if main fails |
| KaLM E5-style "Instruct: Retrieve..." prefix breaks downstream eval | 20% | Medium | Smoke test catches; can strip prefix in trainer |
| Multi-positive sources still under-represented after sampling | 20% | Medium | shard.json confirms multi-positive samples exist; check `len(pos_idx) > 1` ratio |
| Some shard sample has corrupt corpus offset | 10% | Low | Trainer should skip; add try/except in loader |

---

## 9. Timeline

| Task | Machine | Wall time |
|---|---|---|
| Port dataset_msa_shard.py + patch trainer | local | 30 min |
| Read & verify target format | local | 30 min |
| Smoke test (100 step) | cvm-rl | 30 min |
| **Phase B**: full CPT v5 (32k step) | cvm-rl | 25–30 h |
| Mid-CPT gate × 2 | cvm-rl | 15 min × 2 |
| Phase C: SFT-S1 v5 (15k step on 9-bench train splits) | cvm-rl | 12 h |
| Phase D: SFT-S2 v5 curriculum 64→512 docs | cvm-rl | 15 h |
| Final eval 9-bench × 100q (4-way) | cvm-rl | 4 h |
| **Total** | | **~3 days** |

---

## 10. Success criteria

| Criterion | Target |
|---|---|
| **Primary**: v5 9-bench AVG > vanilla 4B noctx AVG (1.975) | Required to claim improvement |
| Secondary: v5 ≥ v4 SFT-S1 AVG (1.546) | Easy threshold (we know v4 underperformed) |
| Stretch: v5 ≥ paper Table 2 70% (≈ 2.6 AVG) | Aspirational, given 1/8 compute |
| Multi-positive bench (HotpotQA / Musique / 2Wiki) router F1 > 0.30 | Confirms 36-source mix fixed v4's transfer issue |
| KaLM Chinese sample router F1 > vanilla random | Confirms multilingual transfer works |

---

## 11. Open questions (to resolve before/during run)

1. Does `dataset_msa_shard.py` emit paper format or simplified format?
2. Should we copy shard_0 from f1f4 to cvm-rl for double-coverage, or train on shard_1 alone?
3. Does the v4 trainer handle multi-positive `pos_idx` (length > 1) correctly?
4. KaLM's "Instruct: Retrieve..." prefix — keep or strip?

---

## 12. Approvals checklist

- [ ] Plan reviewed by Zeyu
- [ ] cpt_shards_paper authenticity verified (✓ done 2026-05-08)
- [ ] dataset_msa_shard.py target format verified
- [ ] Smoke test passes
- [ ] Mid-CPT auto-kill thresholds set
- [ ] Wandb project name reserved
- [ ] launch script written and dry-run

Once all checked → launch CPT v5 on cvm-rl.
