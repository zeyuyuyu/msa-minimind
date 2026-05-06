#!/bin/bash
# Run paper-aligned LLM judge (gemini-2.5-flash via openrouter) on
# bench_evermind_aligned_v2.py JSON outputs.
#
# Usage:
#   bash run_llm_judge.sh <eval_dir>
# Example:
#   bash run_llm_judge.sh /workspace/runs/eval_p5/p5a_step50k_smoke
#
# Requires OPENROUTER_API_KEY env var.

set -e
IN_DIR=${1:?eval_dir required}

if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set" >&2
  exit 1
fi

cd /workspace/msa-minimind

OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
python scripts/llm_judge_evermind.py \
    --in_dir $IN_DIR \
    --backend openrouter \
    2>&1 | tee $IN_DIR/llm_judge.log
echo "[llm_judge] done -> $IN_DIR"
