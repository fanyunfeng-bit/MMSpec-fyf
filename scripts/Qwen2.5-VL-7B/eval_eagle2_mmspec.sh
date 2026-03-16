#!/bin/bash
# EAGLE2 dynamic tree speculative decoding evaluation for MMSpec unified dataset
# Usage: bash scripts/Qwen2.5-VL-7B/eval_eagle2_mmspec.sh [test|testmini]

set -euo pipefail

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SPEC_MODEL="${2:-Cloudriver/EAGLE-Qwen2.5-VL-7B-Instruct}"  # TODO: replace with dedicated EAGLE2 checkpoint when available
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
RESULT_NAME="qwen2.5-vl-7b"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

# EAGLE2 parameters
DEPTH=3
TOP_K=8
TOTAL_TOKEN=30
THRESHOLD="0.3"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Output paths
RESULT_DIR="results/Qwen2.5-VL-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/eagle2-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "EAGLE2 Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Spec Model: ${SPEC_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "EAGLE2 params: depth=${DEPTH}, top_k=${TOP_K}, total_token=${TOTAL_TOKEN}, threshold=${THRESHOLD}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_eagle2_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "eagle2" \
    --temperature "${TEMPERATURE}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --total-token "${TOTAL_TOKEN}" \
    --threshold "${THRESHOLD}" \
    --max-new-token "${MAX_NEW_TOKEN}"

echo ""
echo "EAGLE2 evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
