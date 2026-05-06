#!/bin/bash
# Fire bench_evermind_aligned_v2.py on a given Phase-5 LoRA checkpoint.
# Outputs per-bench JSON (query/pred/gold + router metrics + LM loss + telemetry)
# to /workspace/runs/eval_p5/<run_name>/. No LLM judge call (offline post-process).
#
# Usage:
#   bash run_bench_on_ckpt.sh <ckpt_path> <run_name> [num_queries] [benches]
#
# Example:
#   bash run_bench_on_ckpt.sh /workspace/runs/p5_shard0/p5_shard0_step52000.pt \
#                             p5a_step52k 50 hotpotqa,musique,triviaqa_06M,nature_questions,msmarco_v1

set -e
CKPT=${1:?ckpt path required}
RUN=${2:?run_name required}
NQ=${3:-50}
BENCHES=${4:-hotpotqa,musique,triviaqa_06M,nature_questions,msmarco_v1}

OUT=/workspace/runs/eval_p5/${RUN}
mkdir -p $OUT
echo "[eval] ckpt=$CKPT run=$RUN num_queries=$NQ benches=$BENCHES" | tee $OUT/eval.log

cd /workspace/msa-minimind

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
python -u scripts/bench_evermind_aligned_v2.py \
    --qwen3_5_path /workspace/qwen35_instruct \
    --ckpt $CKPT \
    --bench_root /workspace/msa_bench \
    --encoded_root /workspace/encoded_corpora \
    --out_dir $OUT \
    --benches $BENCHES \
    --num_queries $NQ \
    --top_k 10 \
    --max_new_tokens 1024 \
    --log_every 5 \
    2>&1 | tee -a $OUT/eval.log
echo "[eval] done -> $OUT" | tee -a $OUT/eval.log
