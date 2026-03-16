import argparse
import deepspeed

parser = argparse.ArgumentParser(description='EAGLE3 Stage 2: Multimodal Training')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/llama31chat/8B/')
parser.add_argument('--trainpath', type=str, default="train.json")
parser.add_argument('--testpath', type=str, default="test.json")
parser.add_argument('--imagedir', type=str, default="images/")
parser.add_argument('--savedir', type=str, default='0')
parser.add_argument('--loadpath', type=str, default=None, help="Path to Stage 1 checkpoint")
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json
import re

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 20,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "train/configs/qwen2.5vl_eagle3_config.json",
    "gradient_checkpointing": True
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import os
import torch
from cnets_vlm import padding
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
from train.model.cnets_eagle3_vlm import Model
from train.model.configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup


class MultimodalDataset(Dataset):
    """Dataset for multimodal (image + text) EAGLE3 training."""

    def __init__(self, data_path, image_dir, processor, max_len=2048):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_len = max_len
        # Filter out invalid entries
        self.valid_indices = []
        for i, item in enumerate(self.data):
            if item.get('conversations') and item.get('image'):
                self.valid_indices.append(i)
        print(f"Loaded {len(self.valid_indices)} valid multimodal conversations from {len(self.data)} total")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        item = self.data[self.valid_indices[idx]]

        # Build conversation
        messages = []
        roles = {"human": "user", "gpt": "assistant"}
        source = item['conversations']
        if roles.get(source[0]["from"]) != "user":
            source = source[1:]

        # Load image
        image_path = os.path.join(self.image_dir, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            return None

        for j, sentence in enumerate(source):
            role = roles.get(sentence["from"], sentence["from"])
            content = sentence["value"]
            if role == "user":
                # Replace <image> token with Qwen2.5-VL image reference
                if "<image>" in content:
                    content = content.replace("<image>", "")
                    messages.append({
                        "role": role,
                        "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": content.strip()},
                        ]
                    })
                else:
                    messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role, "content": content})

        # Process with Qwen2.5-VL processor
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Do NOT truncate — truncation can cut image placeholder tokens
            # while keeping pixel_values intact, causing token/feature mismatch
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
        except Exception:
            return None

        input_ids = inputs["input_ids"].squeeze(0)
        # Skip samples that are too long (instead of truncating, which breaks image token alignment)
        if len(input_ids) > self.max_len:
            return None

        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        # Create loss mask: only train on assistant responses
        loss_mask = torch.zeros_like(input_ids)
        sep_assistant = "<|im_start|>assistant\n"
        sep_end = "<|im_end|>"

        # Decode to find assistant regions
        decoded = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        parts = decoded.split(sep_assistant)
        cur_pos = 0
        for pi, part in enumerate(parts):
            if pi == 0:
                cur_pos += len(self.processor.tokenizer(part + sep_assistant, add_special_tokens=False).input_ids)
            else:
                if sep_end in part:
                    response = part.split(sep_end)[0]
                    resp_len = len(self.processor.tokenizer(response, add_special_tokens=False).input_ids)
                    end_pos = min(cur_pos + resp_len, len(loss_mask))
                    loss_mask[cur_pos:end_pos] = 1
                cur_pos += len(self.processor.tokenizer(
                    part + (sep_assistant if pi < len(parts) - 1 else ""),
                    add_special_tokens=False
                ).input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def collate_multimodal(features):
    """Custom collate function that handles variable-size pixel_values and skips None samples."""
    # Filter out None samples
    features = [f for f in features if f is not None]
    if len(features) == 0:
        return None

    # For batch_size=1, just unsqueeze
    # For larger batches, need to pad input_ids to same length
    max_length = max(f['input_ids'].shape[0] for f in features)

    batch_input_ids = []
    batch_attention_mask = []
    batch_loss_mask = []
    batch_pixel_values = []
    batch_image_grid_thw = []

    for f in features:
        seq_len = f['input_ids'].shape[0]
        pad_len = max_length - seq_len

        if pad_len > 0:
            batch_input_ids.append(torch.cat([f['input_ids'], torch.zeros(pad_len, dtype=f['input_ids'].dtype)]))
            batch_attention_mask.append(torch.cat([f['attention_mask'], torch.zeros(pad_len, dtype=f['attention_mask'].dtype)]))
            batch_loss_mask.append(torch.cat([f['loss_mask'], torch.zeros(pad_len, dtype=f['loss_mask'].dtype)]))
        else:
            batch_input_ids.append(f['input_ids'])
            batch_attention_mask.append(f['attention_mask'])
            batch_loss_mask.append(f['loss_mask'])

        if f['pixel_values'] is not None:
            batch_pixel_values.append(f['pixel_values'])
        if f['image_grid_thw'] is not None:
            batch_image_grid_thw.append(f['image_grid_thw'])

    batch = {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "loss_mask": torch.stack(batch_loss_mask),
    }

    if batch_pixel_values:
        batch["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
    if batch_image_grid_thw:
        batch["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)

    return batch


# Load config and pre-compute cache.pt before model loading
config = EConfig.from_pretrained(train_config["config_path"])

# cache.pt should already exist from Stage 1
if not os.path.exists("cache.pt"):
    print("WARNING: cache.pt not found. Run Stage 1 first to generate it.")
    print("Using Stage 1's cache.pt or generating a new one from text data...")

# Load processor and build datasets
processor = AutoProcessor.from_pretrained(args.basepath)

traindataset = MultimodalDataset(args.trainpath, args.imagedir, processor, max_len=train_config["max_len"])
# Use smaller test set or same data for test
if args.testpath and os.path.exists(args.testpath):
    testdataset = MultimodalDataset(args.testpath, args.imagedir, processor, max_len=train_config["max_len"])
else:
    testdataset = None

# config already created above
model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True)
model.scandata(args.trainpath if not args.trainpath.endswith('.json') else args.trainpath, args.basepath)

# Load Stage 1 checkpoint weights into draft model
if args.loadpath and os.path.exists(args.loadpath):
    print(f"Loading Stage 1 weights from {args.loadpath}")
    # Load all safetensors/bin files from the checkpoint directory
    import glob
    st_files = glob.glob(os.path.join(args.loadpath, "*.safetensors"))
    if st_files:
        state_dict = {}
        for sf in st_files:
            with safe_open(sf, framework="pt") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded Stage 1 checkpoint: {len(state_dict)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")
    else:
        pt_files = glob.glob(os.path.join(args.loadpath, "*.bin")) + glob.glob(os.path.join(args.loadpath, "*.pt"))
        if pt_files:
            state_dict = {}
            for pf in pt_files:
                sd = torch.load(pf, map_location="cpu")
                state_dict.update(sd)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded Stage 1 checkpoint: {len(state_dict)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")
        else:
            print(f"No checkpoint files found in {args.loadpath}")
else:
    print("No Stage 1 checkpoint specified, training from scratch")


criterion = nn.SmoothL1Loss(reduction="none")
num_epochs = train_config["num_epochs"]

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     )

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
if global_rank == 0:
    print("EAGLE3 Stage 2: Multimodal training for Qwen2.5-VL")

os.makedirs(args.savedir, exist_ok=True)

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                          pin_memory=True, collate_fn=collate_multimodal)

if testdataset is not None:
    test_sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=test_sampler, num_workers=4,
                             pin_memory=True, collate_fn=collate_multimodal)
else:
    test_loader = None


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
if checkpoint_path:
    print(f"Resuming from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)


for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch + 1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(train_loader)):
        if data is None:
            continue

        model.zero_grad()

        # Prepare multimodal inputs
        kwargs = {
            "input_ids": data["input_ids"].to(rank),
            "attention_mask": data["attention_mask"].to(rank),
            "loss_mask": data["loss_mask"],
        }
        if "pixel_values" in data:
            kwargs["pixel_values"] = data["pixel_values"].to(rank)
        if "image_grid_thw" in data:
            kwargs["image_grid_thw"] = data["image_grid_thw"].to(rank)

        try:
            plosses, vlosses, acces = model_engine(**kwargs)
        except (ValueError, RuntimeError) as e:
            if global_rank == 0 and batch_idx % 1000 == 0:
                print(f"Skipping batch {batch_idx}: {e}")
            continue

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        model_engine.backward(loss)
        model_engine.step()

        if global_rank == 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            if batch_idx % 100 == 0:
                print(f"Step {batch_idx}: lr={logdict['train/lr']:.2e}, ploss_0={plosses[0].item():.4f}, acc_0={acces[0]:.4f}")
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

        # Save mid-epoch checkpoint every 5000 steps to avoid losing progress on timeout
        if batch_idx > 0 and batch_idx % 5000 == 0:
            if global_rank == 0:
                print(f"Saving mid-epoch checkpoint at epoch {epoch}, step {batch_idx}...")
            model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}_step{batch_idx}", exclude_frozen_parameters=True)

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    # Evaluation
    if test_loader is not None:
        epoch_acces = [[] for _ in range(model.length)]
        epoch_plosses = [[] for _ in range(model.length)]

        for batch_idx, data in enumerate(tqdm(test_loader)):
            if data is None:
                continue
            with torch.no_grad():
                kwargs = {
                    "input_ids": data["input_ids"].to(rank),
                    "attention_mask": data["attention_mask"].to(rank),
                    "loss_mask": data["loss_mask"],
                }
                if "pixel_values" in data:
                    kwargs["pixel_values"] = data["pixel_values"].to(rank)
                if "image_grid_thw" in data:
                    kwargs["image_grid_thw"] = data["image_grid_thw"].to(rank)

                plosses, vlosses, acces = model_engine(**kwargs)
                epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
                epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
            acc_i = acc_i.item()
            if global_rank == 0:
                print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
            loss_i = loss_i.item()
            if global_rank == 0:
                print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    torch.cuda.empty_cache()

    model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if epoch % 5 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")
