"""
Data generation script for EAGLE-1 training on LLaVA-1.5.

This script extracts second-to-last hidden states and saves them as .pt files.
For multimodal stage-2 training, pass --save_inputs_embeds to also store fused
multimodal input embeddings from hidden_states[0].
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm


def load_model_and_processor(model_path):
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def load_data(datapath, start=0, end=None):
    with open(datapath, "r") as f:
        first_char = ""
        while True:
            c = f.read(1)
            if not c:
                break
            if not c.isspace():
                first_char = c
                break
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if end is None:
        end = len(data)
    return data[start:end]


def _normalize_conv_text(text):
    return text.replace("<image>\n", "<image>\n").strip()


def build_prompt(sample):
    lines = []
    for turn in sample.get("conversations", []):
        role = turn.get("from", "human")
        value = _normalize_conv_text(turn.get("value", ""))
        if role in ["human", "user"]:
            lines.append(f"USER: {value}")
        else:
            lines.append(f"ASSISTANT: {value}")
    return "\n".join(lines)


def resolve_image(sample, image_base_dir=None):
    image_path = sample.get("image", "")
    if not image_path:
        return None
    if image_base_dir and not os.path.isabs(image_path):
        image_path = os.path.join(image_base_dir, image_path)
    if not os.path.exists(image_path):
        return None
    return Image.open(image_path).convert("RGB")


@torch.no_grad()
def generate_data(model, processor, sample, image_base_dir=None, max_length=2048, save_inputs_embeds=False):
    try:
        prompt = build_prompt(sample)
        image = resolve_image(sample, image_base_dir)

        if image is not None:
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        else:
            inputs = processor(text=prompt, return_tensors="pt")

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        if inputs["input_ids"].shape[1] > max_length:
            return None

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_state = outputs.hidden_states[-2].squeeze(0).cpu()
        input_ids = inputs["input_ids"].squeeze(0).cpu()
        loss_mask = torch.ones_like(input_ids)

        result = {
            "hidden_state": hidden_state,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }

        if save_inputs_embeds:
            result["inputs_embeds"] = outputs.hidden_states[0].squeeze(0).cpu()

        return result
    except Exception as e:
        print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate EAGLE training data for LLaVA-1.5")
    parser.add_argument("--model", type=str, required=True, help="Path to llava-v1.5-7b-hf model")
    parser.add_argument("--datapath", type=str, required=True, help="Path to training data JSONL")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--image_base_dir", type=str, default=None, help="Base directory for images")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "--save_inputs_embeds",
        action="store_true",
        help="Save fused multimodal inputs_embeds for stage-2 training",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading model from {args.model}...")
    model, processor = load_model_and_processor(args.model)

    print(f"Loading data from {args.datapath}...")
    data = load_data(args.datapath, args.start, args.end)
    print(f"Loaded {len(data)} samples")

    saved_count = 0
    for i, sample in enumerate(tqdm(data, desc="Generating data")):
        result = generate_data(
            model,
            processor,
            sample,
            image_base_dir=args.image_base_dir,
            max_length=args.max_length,
            save_inputs_embeds=args.save_inputs_embeds,
        )
        if result is None:
            continue
        idx = args.start + i
        outpath = os.path.join(args.outdir, f"{idx}.pt")
        torch.save(result, outpath)
        saved_count += 1

    print(f"Saved {saved_count} samples to {args.outdir}")


if __name__ == "__main__":
    main()
