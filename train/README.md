# Training Code

This directory consolidates training code for EAGLE (Stage 1/2), EAGLE3, Medusa, and MSD so it can be run from the project root.

## Directory Structure

- `main_stage1.py`: EAGLE 1/2 Stage 1 (text-only) training using Accelerate.
- `main_stage2.py`: EAGLE 1/2 Stage 2 (multimodal) training using Accelerate.
- `main_eagle3_stage1.py`: EAGLE 3 Stage 1 (text-only) training using DeepSpeed.
- `main_eagle3_stage2.py`: EAGLE 3 Stage 2 (multimodal) training using DeepSpeed.
- `msd/`: MSD training stack migrated from `MSD/EAGLE/eagle/` (entrypoint, configs, model code).
- `model/`: Contains all draft model architectures (`cnets.py`, `cnets_multimodal.py`, `cnets_eagle3.py`) and modeling utilities.
- `configs/`: JSON configuration files for the draft models.
- `data/`: Scripts for data generation (e.g., extracting hidden states).
- `scripts/`: Ready-to-use Bash and SLURM (`sbatch`) scripts for kicking off training jobs.
  - Includes Qwen2.5-VL scripts (`run_eagle_stage1.sh`, `run_eagle_stage2.sh`) and
    LLaVA-1.5-7B scripts (`run_eagle_stage1_llava.sh`, `run_eagle_stage2_llava.sh`).
- `ds_config*.json`: DeepSpeed configurations for EAGLE 1/2/3.

## How to Run

**Important:** Always run the training scripts from the **project root** (`/fs/scratch/PAS2136/ziheng/mmspec`).

### Using Helper Scripts (Recommended)

All shell and SLURM scripts are located in `train/scripts/`.

Example running EAGLE 3 Stage 1 via SLURM:
```bash
sbatch train/scripts/run_eagle3_stage1.sbatch
```

Example running EAGLE 1/2 Stage 2 interactively:
```bash
bash train/scripts/run_eagle_stage2.sh
```

Example running MSD (Qwen2.5-VL) with DeepSpeed:
```bash
bash train/scripts/run_msd_qwen25vl.sh
```

### Manual Execution

**EAGLE 1/2 (Accelerate):**
```bash
accelerate launch --multi_gpu --mixed_precision=bf16 \
    -m train.main_stage1 \
    --basepath <model_path> ...
```

**LLaVA-1.5-7B data generation (for EAGLE1):**
```bash
# stage1 text-only data
python -m train.data.ge_data_llava \
  --model <llava-v1.5-7b-hf> \
  --datapath <text_train.jsonl> \
  --outdir <stage1_pt_dir>

# stage2 multimodal data (save inputs_embeds)
python -m train.data.ge_data_llava \
  --model <llava-v1.5-7b-hf> \
  --datapath <mm_train.jsonl> \
  --image_base_dir <images_dir> \
  --outdir <stage2_pt_dir> \
  --save_inputs_embeds
```

**EAGLE 3 (DeepSpeed):**
```bash
deepspeed -m train.main_eagle3_stage2 \
    --basepath <model_path> ...
```

**MSD (DeepSpeed):**
```bash
deepspeed -m train.msd.main_deepspeed \
    --deepspeed_config train/msd/ds_config.json \
    --tmpdir_v <multimodal_pt_dir> \
    --tmpdir_t <text_pt_dir> \
    --basepath <Qwen_or_LLaVA_base_model> \
    --cpdir <checkpoint_output_dir> \
    --config train/msd/qwen2vl_config.json
```
