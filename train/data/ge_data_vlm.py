"""
Data generation script for EAGLE-1 VLM training with Qwen2.5-VL.

This script:
1. Loads Qwen2.5-VL model
2. Processes image+text data
3. Extracts second-to-last layer hidden states
4. Saves as .pt files for EAGLE training

Usage:
    python -m eagle.data_vlm.ge_data_vlm \
        --model /path/to/Qwen2.5-VL-7B-Instruct \
        --datapath /path/to/train.jsonl \
        --outdir /path/to/output \
        --start 0 --end 1000
"""

import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info


def load_model(model_path, device="cuda"):
    """Load Qwen2.5-VL model and processor."""
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.eval()
    
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    
    return model, processor


def load_data(datapath, start=0, end=None):
    """Load JSONL data file."""
    data = []
    with open(datapath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if end is None:
        end = len(data)
    return data[start:end]


def build_messages(sample, image_base_dir=None):
    """Build messages for Qwen2.5-VL from sample.
    
    Expected sample format:
    {
        "id": "...",
        "image": "/path/to/image.jpg" or "relative/path.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nQuestion..."},
            {"from": "gpt", "value": "Answer..."}
        ]
    }
    """
    messages = []
    
    for turn in sample.get("conversations", []):
        role = "user" if turn["from"] in ["human", "user"] else "assistant"
        content = turn["value"]
        
        if role == "user" and "<image>" in content:
            # Replace <image> with actual image reference
            image_path = sample.get("image", "")
            if image_base_dir and not os.path.isabs(image_path):
                image_path = os.path.join(image_base_dir, image_path)
            
            # Build content with image
            content_list = []
            parts = content.split("<image>")
            for i, part in enumerate(parts):
                if i > 0:  # Add image before this part
                    content_list.append({
                        "type": "image",
                        "image": f"file://{image_path}"
                    })
                if part.strip():
                    content_list.append({
                        "type": "text",
                        "text": part
                    })
            
            messages.append({"role": role, "content": content_list})
        else:
            messages.append({"role": role, "content": content})
    
    return messages


@torch.no_grad()
def generate_data(model, processor, sample, image_base_dir=None, max_length=2048):
    """Generate hidden states for a single sample."""
    try:
        messages = build_messages(sample, image_base_dir)
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Tokenize
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # Truncate if too long
        if inputs.input_ids.shape[1] > max_length:
            return None
        
        # Forward pass with hidden states
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get second-to-last layer hidden states (EAGLE-1 uses this)
        # hidden_states is a tuple of (embeddings, layer1, layer2, ..., final_layer)
        # Second-to-last is hidden_states[-2]
        hidden_state = outputs.hidden_states[-2].squeeze(0)  # [seq_len, hidden_dim]
        input_ids = inputs.input_ids.squeeze(0)  # [seq_len]
        
        # Create loss mask (1 for assistant tokens, 0 for user/system tokens)
        # For simplicity, we'll create a basic loss mask
        # In practice, you should properly identify assistant response spans
        loss_mask = torch.ones_like(input_ids)
        
        return {
            "hidden_state": hidden_state.cpu(),
            "input_ids": input_ids.cpu(),
            "loss_mask": loss_mask.cpu()
        }
        
    except Exception as e:
        print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate EAGLE training data for VLM")
    parser.add_argument("--model", type=str, required=True, help="Path to Qwen2.5-VL model")
    parser.add_argument("--datapath", type=str, required=True, help="Path to training data JSONL")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--image_base_dir", type=str, default=None, help="Base directory for images")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, processor = load_model(args.model)
    
    # Load data
    print(f"Loading data from {args.datapath}...")
    data = load_data(args.datapath, args.start, args.end)
    print(f"Loaded {len(data)} samples")
    
    # Generate data
    saved_count = 0
    for i, sample in enumerate(tqdm(data, desc="Generating data")):
        result = generate_data(
            model, processor, sample, 
            image_base_dir=args.image_base_dir,
            max_length=args.max_length
        )
        
        if result is not None:
            # Save to file
            idx = args.start + i
            outpath = os.path.join(args.outdir, f"{idx}.pt")
            torch.save(result, outpath)
            saved_count += 1
    
    print(f"Saved {saved_count} samples to {args.outdir}")


if __name__ == "__main__":
    main()
