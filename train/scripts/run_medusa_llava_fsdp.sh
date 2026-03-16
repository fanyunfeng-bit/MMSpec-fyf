#!/bin/bash
# =============================================================================
# Medusa FSDP Multi-Node Training (LLaVA-1.5-7B)
# Matching ViSpec_ping hyperparameters
# =============================================================================
#
# Usage:  On each of the 8 nodes, run:
#
#   bash train/scripts/run_medusa_llava_fsdp.sh <NODE_RANK> <MASTER_ADDR>
#
#   NODE_RANK:   0-7  (unique per node, node 0 is master)
#   MASTER_ADDR: hostname of node 0 (e.g. a0130)
#
# Example (8 terminals on nodes a0130..a0137):
#   [a0130] bash train/scripts/run_medusa_llava_fsdp.sh 0 a0130
#   [a0131] bash train/scripts/run_medusa_llava_fsdp.sh 1 a0130
#   [a0132] bash train/scripts/run_medusa_llava_fsdp.sh 2 a0130
#   ...
#   [a0137] bash train/scripts/run_medusa_llava_fsdp.sh 7 a0130
#
# Prerequisites:
#   1) Hidden states extracted via:
#      python -m train.data.allocation_llava_shargpt \
#          --outdir <OUTDIR> --start 0 --end 67999 \
#          --model llava-hf/llava-1.5-7b-hf
#
#   2) Set DATADIR below to the extraction output path
# =============================================================================

set -e

NODE_RANK=${1:?"Usage: $0 <NODE_RANK 0-7> <MASTER_ADDR>"}
MASTER_ADDR=${2:?"Usage: $0 <NODE_RANK 0-7> <MASTER_ADDR>"}
MASTER_PORT=${3:-29500}

# ---- NCCL / Distributed hardening ----
export NCCL_DEBUG=WARN                          # INFO for full debug, WARN for less noise
export NCCL_TIMEOUT=1800000                     # 30 min (ms), up from default 600s
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000        # Enable flight recorder for better stack traces
export NCCL_IB_TIMEOUT=23                       # InfiniBand timeout (default ~20)
export NCCL_SOCKET_IFNAME=^lo,docker            # Exclude loopback/docker interfaces
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce CUDA memory fragmentation

# ---- Paths (EDIT THESE) ----
BASEPATH="llava-hf/llava-1.5-7b-hf"
CONFIGPATH="train/configs/llava1.5_7b_config.json"
DATADIR="/users/PAS2099/ping69852/ps/Multimodal-Speculative-Decoding-Benchmark/data/sharegpt_processed/llava_shargpt_0_67999_mufp16"   # <-- hidden state dir
CPDIR="/users/PAS2099/ping69852/ps/Multimodal-Speculative-Decoding-Benchmark/checkpoints_medusa_llava"          # <-- checkpoint output

# ---- Hyperparameters (matching ViSpec_ping) ----
LR=3e-5
BS=1                         # per-GPU batch size
GRAD_ACCUM=1                 # effective batch = 8 GPUs × 1 × 1 = 8 (matches paper)
MAX_LEN=2048                 # matches ViSpec_ping default
NUM_WORKERS=2                # matches ViSpec_ping default

echo "============================================"
echo " Medusa FSDP Training — Node ${NODE_RANK}/7"
echo " Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo " Model:  ${BASEPATH}"
echo " Data:   ${DATADIR}"
echo " Ckpt:   ${CPDIR}"
echo "============================================"

accelerate launch \
    --config_file train/configs/accelerate_fsdp_8node.yaml \
    --machine_rank "${NODE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    -m train.main_medusa \
    --basepath "${BASEPATH}" \
    --configpath "${CONFIGPATH}" \
    --tmpdir "${DATADIR}" \
    --cpdir "${CPDIR}" \
    --bs ${BS} \
    --gradient-accumulation-steps ${GRAD_ACCUM} \
    --num-workers ${NUM_WORKERS} \
    --lr ${LR} \
    --max-len ${MAX_LEN} \
    2>&1 | tee train_llava_log_node${NODE_RANK}.txt
