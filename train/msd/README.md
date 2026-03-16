# MSD Training (Integrated)

This directory contains the MSD training code used in this project, reorganized from:

- `MSD/EAGLE/eagle/train/main_deepspeed.py`
- `MSD/EAGLE/eagle/model/*`

The code was copied into `train/msd/` so MSD training can be managed from the unified `train/` tree.

## Files

- `main_deepspeed.py`: MSD training entrypoint (DeepSpeed).
- `ds_config.json`: DeepSpeed config used by MSD training.
- `qwen2vl_config.json`: Qwen2-VL draft model config for MSD.
- `llava_v15_7B_config.json`, `llava_v15_13B_config.json`, `llama_2_chat_7B_config.json`: legacy configs retained for compatibility.
- `model/`: MSD model code required by `main_deepspeed.py`.

## Run

From project root:

```bash
bash train/scripts/run_msd_qwen25vl.sh
```

or directly:

```bash
deepspeed -m train.msd.main_deepspeed \
  --deepspeed_config train/msd/ds_config.json \
  --tmpdir_v <multimodal_pt_dir> \
  --tmpdir_t <text_pt_dir> \
  --basepath <Qwen/Qwen2.5-VL-7B-Instruct_or_local_path> \
  --cpdir <checkpoint_output_dir> \
  --config train/msd/qwen2vl_config.json
```
