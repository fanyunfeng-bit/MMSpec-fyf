#!/bin/bash
# ViSpec-vskip evaluation for MMSpec unified dataset
# Usage: bash scripts/LLaVA-1.5-7B/eval_vispec_vskip_mmspec.sh [test|testmini]

set -euo pipefail

BASE_MODEL="llava-hf/llava-1.5-7b-hf"
SPEC_MODEL="Cloudriver/ViSpec-LLaVA1.5-7B"
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

DEPTH=3
TOP_K=8
TOTAL_TOKEN=30
NUM_Q=2
VISUAL_ATTN_THRESHOLD="${VISUAL_ATTN_THRESHOLD:-0.35}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

RESULT_DIR="results/LLaVA-1.5-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/vispec_vskip-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "ViSpec-vskip Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Spec Model: ${SPEC_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "ViSpec-vskip params: depth=${DEPTH}, top_k=${TOP_K}, total_token=${TOTAL_TOKEN}, num_q=${NUM_Q}, visual_attn_threshold=${VISUAL_ATTN_THRESHOLD}"
echo "Attention implementation: ${ATTN_IMPLEMENTATION}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_vispec_vskip_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "vispec_vskip" \
    --temperature "${TEMPERATURE}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --total-token "${TOTAL_TOKEN}" \
    --num-q "${NUM_Q}" \
    --visual-attn-threshold "${VISUAL_ATTN_THRESHOLD}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --max-new-token "${MAX_NEW_TOKEN}"

echo ""
echo "ViSpec-vskip evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
