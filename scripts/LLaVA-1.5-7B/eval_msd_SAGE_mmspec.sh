#!/bin/bash
# MSD + SAGE (visual-token compression on the draft) evaluation for MMSpec.
# Usage: bash scripts/LLaVA-1.5-7B/eval_msd_SAGE_mmspec.sh [test|testmini]
#   OVERWRITE=1 to delete an existing answer file before running.

set -euo pipefail

BASE_MODEL="llava-hf/llava-1.5-7b-hf"
MSD_MODEL="${2:-Cloudriver/MSD-LLaVA1.5-7B}"
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

# MSD parameters
TOTAL_TOKEN=-1
DEPTH=5
TOP_K=10
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

# SAGE parameters
SAGE_THRESHOLD_MODE="abs"          # abs | topk_ratio
SAGE_THRESHOLD_VALUE=100
SAGE_MIN_SINKS=0
SAGE_ENABLE_COMPRESSOR="true"
SAGE_COMPRESSOR_LAYER=16
SAGE_COMPRESSOR_TOPK=10

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

RESULT_DIR="results/LLaVA-1.5-7B/mmspec_${SPLIT}"
mkdir -p "${RESULT_DIR}"

ANSWER_FILE="${RESULT_DIR}/msd_SAGE-${SAGE_THRESHOLD_MODE}-${SAGE_THRESHOLD_VALUE}-k${SAGE_COMPRESSOR_TOPK}-temperature-${TEMPERATURE}.jsonl"

if [[ "${OVERWRITE:-0}" == "1" ]]; then
    rm -f "${ANSWER_FILE}"
fi

echo "=========================================="
echo "MSD + SAGE Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "MSD Model: ${MSD_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "MSD params: depth=${DEPTH}, top_k=${TOP_K}, total_token=${TOTAL_TOKEN}"
echo "SAGE params: mode=${SAGE_THRESHOLD_MODE}, value=${SAGE_THRESHOLD_VALUE}, "\
"layer=${SAGE_COMPRESSOR_LAYER}, topk=${SAGE_COMPRESSOR_TOPK}"
echo "Attention implementation: ${ATTN_IMPLEMENTATION}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

python -m evaluation.eval_msd_SAGE_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --msd-model-path "${MSD_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "msd_SAGE" \
    --temperature "${TEMPERATURE}" \
    --use-msd \
    --total-token "${TOTAL_TOKEN}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --max-new-token "${MAX_NEW_TOKEN}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --sage-threshold-mode "${SAGE_THRESHOLD_MODE}" \
    --sage-threshold-value "${SAGE_THRESHOLD_VALUE}" \
    --sage-min-sinks "${SAGE_MIN_SINKS}" \
    --sage-enable-compressor "${SAGE_ENABLE_COMPRESSOR}" \
    --sage-compressor-layer "${SAGE_COMPRESSOR_LAYER}" \
    --sage-compressor-topk "${SAGE_COMPRESSOR_TOPK}" \
    --sage-debug

echo ""
echo "MSD + SAGE evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
