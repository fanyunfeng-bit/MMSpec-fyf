import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/vicuna_v13/7B/')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--loadpath', type=str, default=None)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--max-len', type=int, default=2048)
parser.add_argument('--tmpdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument('--pw', type=float, default=0.1)
parser.add_argument('--begin-epoch', type=int, default=0)
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": args.pw,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": args.num_workers,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": args.max_len,
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5
}
import json
from safetensors import safe_open
import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
)
from train.model.cnets_medusa import Model
from train.model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig, AutoModelForImageTextToText

if accelerator.is_main_process:
    try:
        import wandb
        wandb.init(project="ess-medusa", config=train_config)
    except ModuleNotFoundError:
        wandb = None
else:
    wandb = None


def wandb_log(logdict):
    if accelerator.is_main_process and wandb is not None:
        wandb.log(logdict)

def log_main(msg):
    """Print only on main process with timestamp."""
    if accelerator.is_main_process:
        import datetime
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {msg}", flush=True)

# ============ STARTUP LOGGING ============
log_main("="*70)
log_main("MEDUSA TRAINING — STARTUP")
log_main("="*70)
log_main(f"Distributed: num_processes={accelerator.num_processes}, device={accelerator.device}")
log_main(f"Process index: {accelerator.process_index}, local: {accelerator.local_process_index}")
log_main(f"Mixed precision: {accelerator.mixed_precision}")
log_main(f"Gradient accumulation steps: {train_config['gradient_accumulation_steps']}")
log_main("--- Hyperparameters ---")
for k, v in sorted(train_config.items()):
    log_main(f"  {k}: {v}")
log_main("--- CLI Args ---")
for k, v in sorted(vars(args).items()):
    log_main(f"  {k}: {v}")
log_main("="*70)

baseconfig = AutoConfig.from_pretrained(args.basepath)

try:
    head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)
except:
    head = torch.nn.Linear(
        baseconfig.text_config.hidden_size,
        baseconfig.text_config.vocab_size,
        bias=False,
    )

try:
    try:
        with open(
            os.path.join(args.basepath, "model.safetensors.index.json"), "r"
        ) as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(
            os.path.join(args.basepath, head_path), framework="pt", device="cpu"
        ) as f:
            tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        with open(
            os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r"
        ) as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path))
        tensor = weights["lm_head.weight"].float()
except:
    m = AutoModelForImageTextToText.from_pretrained(args.basepath, torch_dtype="auto")
    try:
        tensor = m.language_model.lm_head.weight.float()
    except:
        tensor = m.lm_head.weight.float()
    del m

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False

log_main(f"Loaded lm_head: shape={tensor.shape}, dtype={tensor.dtype}, norm={tensor.norm().item():.4f}")


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            if not file.endswith(".ckpt"):
                continue
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data["hidden_state"][: train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][: train_config["max_len"]][None, :]

        length = hidden_state.shape[1]
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["hidden_state_big"].shape[1] for item in features)
        batch_hidden_states = torch.cat(
            [
                self.paddingtensor(item["hidden_state_big"], max_length)
                for item in features
            ]
        )
        batch_target = torch.cat(
            [self.paddingtensor(item["target"], max_length) for item in features]
        )
        batch_loss_mask = torch.tensor(
            [
                item["loss_mask"] + [0] * (max_length - len(item["loss_mask"]))
                for item in features
            ]
        )
        batch = {
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def compute_loss(target, target_p, predict, loss_mask):
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, ploss, out_head


if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])
log_main(f"Total .ckpt files found: {len(datapath)}")
if len(datapath) == 0:
    raise RuntimeError(f"No .ckpt files found in {train_config['datapath']}!")

traindatapath = datapath[: int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95) :]
log_main(f"Train samples: {len(traindatapath)}, Test samples: {len(testdatapath)}")

traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)

train_loader = DataLoader(
    traindataset,
    batch_size=train_config["bs"],
    shuffle=True,
    collate_fn=DataCollatorWithPadding(),
    num_workers=train_config["num_workers"],
    pin_memory=True,
)
test_loader = DataLoader(
    testdataset,
    batch_size=train_config["bs"],
    shuffle=False,
    collate_fn=DataCollatorWithPadding(),
    num_workers=train_config["num_workers"],
    pin_memory=True,
)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])
log_main(f"EConfig: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
model = Model(config, load_emb=True, path=args.basepath)

# Log model parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
log_main(f"Model params: total={total_params:,}, trainable={trainable_params:,}, frozen={frozen_params:,}")
for name, p in model.named_parameters():
    log_main(f"  {'[TRAIN]' if p.requires_grad else '[FROZE]'} {name}: {list(p.shape)}, norm={p.data.norm().item():.4f}")

if args.loadpath:
    from safetensors.torch import load
    with open(args.loadpath, "rb") as f:
        state_dict = load(f.read())
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            print(f"missing_keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"unexpected_keys: {unexpected_keys}")

criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))
log_main(f"Optimizer: AdamW, lr={train_config['lr']}, betas=({train_config['b1']}, {train_config['b2']})")

import math

num_epochs = train_config["num_epochs"]
# num_warmup_steps must be in optimizer-step units (scheduler.step() is called once
# per gradient_accumulation_steps batches), so divide batch count by ga_steps.
num_warmup_steps = math.ceil(len(train_loader) / max(1, train_config["gradient_accumulation_steps"]))
total_steps = math.ceil(len(train_loader) / max(1, train_config["gradient_accumulation_steps"])) * num_epochs
is_warmup = train_config["is_warmup"]

log_main(f"Scheduler: linear warmup, warmup_steps={num_warmup_steps}, total_steps={total_steps}")
log_main(f"Train loader: {len(train_loader)} batches/epoch, Test loader: {len(test_loader)} batches")
log_main(f"Effective batch size: {train_config['bs']} * {accelerator.num_processes} * {train_config['gradient_accumulation_steps']} = {train_config['bs'] * accelerator.num_processes * train_config['gradient_accumulation_steps']}")

# Move frozen head to device without distributed wrapping (for FSDP compatibility)
head = head.to(accelerator.device)
log_main(f"head moved to {accelerator.device} (frozen, NOT wrapped by accelerator)")

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

log_main(f"accelerator.prepare() done.")
log_main("="*70)

# Global step counter for logging
global_step = 0
start_epoch = args.begin_epoch

# ============ AUTO-RESUME from latest checkpoint ============
def find_latest_checkpoint(cpdir):
    """Scan cpdir for state_N directories and return (latest_epoch, path) or (None, None)."""
    if not os.path.exists(cpdir):
        return None, None
    max_epoch = -1
    best_path = None
    for name in os.listdir(cpdir):
        if name.startswith("state_") and os.path.isdir(os.path.join(cpdir, name)):
            try:
                ep = int(name.split("_")[1])
                if ep > max_epoch:
                    max_epoch = ep
                    best_path = os.path.join(cpdir, name)
            except (ValueError, IndexError):
                continue
    if max_epoch >= 0:
        return max_epoch, best_path
    return None, None

resume_epoch, resume_path = find_latest_checkpoint(args.cpdir)
if resume_path is not None:
    log_main(f"[RESUME] Found checkpoint: {resume_path} (epoch {resume_epoch})")
    # Load resume metadata (global_step)
    resume_meta_path = os.path.join(resume_path, "resume_meta.json")
    if os.path.exists(resume_meta_path):
        with open(resume_meta_path, "r") as f:
            resume_meta = json.load(f)
        global_step = resume_meta.get("global_step", 0)
        log_main(f"[RESUME] Restored global_step={global_step}")
    # accelerator.load_state restores model, optimizer, scheduler, RNG states
    accelerator.load_state(resume_path)
    start_epoch = resume_epoch + 1
    log_main(f"[RESUME] Will continue from epoch {start_epoch} (loaded state from epoch {resume_epoch})")
else:
    log_main(f"[RESUME] No existing checkpoint found in {args.cpdir}, training from scratch.")

log_main(f"Starting training from epoch {start_epoch} to {num_epochs-1}")
log_main("="*70)

for epoch in range(start_epoch, num_epochs):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    epoch_vloss = 0
    epoch_ploss = 0
    num_batches = 0
    model.train()
    log_main(f"--- Epoch {epoch}/{num_epochs} TRAIN start ---")
    import time as _time
    _epoch_start = _time.time()
    for batch_idx, data in enumerate(
        tqdm(train_loader, disable=not accelerator.is_local_main_process)
    ):

        with accelerator.accumulate(model):
            hidden_states = data["hidden_states"]
            B = hidden_states.shape[0]
            predict_b = []
            for b in range(B):
                pred_b = model(hidden_states[b : b + 1])
                predict_b.append(pred_b)
            predict = torch.stack(predict_b, dim=1)

            K = predict.shape[0]
            # predict: [K, B, S, H] — keep per-head to avoid [K*B, S, V] OOM

            # Build shifted targets per head
            tgt = []
            for i in range(K):
                with torch.no_grad():
                    shifted = torch.cat(
                        (
                            data["target"][:, i:],
                            torch.zeros_like(data["target"][:, :i]),
                        ),
                        dim=1,
                    )
                tgt.append(shifted)

            loss_mask_base = data["loss_mask"][:, :, None]  # [B, S, 1]

            # --- Detailed shape/value logging (first 5 batches of each epoch) ---
            if accelerator.is_main_process and batch_idx < 5:
                _S = data["target"].shape[1]
                _H = data["target"].shape[2]
                _mask_sum = loss_mask_base.sum().item()
                log_main(f"[TRAIN batch={batch_idx}] B={B}, S={_S}, H={_H}, K={K}, mask_sum={_mask_sum:.0f}")
                log_main(f"  predict shape={list(predict.shape)}, dtype={predict.dtype}, requires_grad={predict.requires_grad}")
                log_main(f"  hidden_states norm={data['hidden_states'].norm().item():.2f}, target norm={data['target'].norm().item():.2f}")
                for ki in range(K):
                    log_main(f"  tgt[{ki}] norm={tgt[ki].norm().item():.2f}, pred[{ki}] norm={predict[ki].norm().item():.2f}")

                # === SHIFT VERIFICATION: check target tokens differ across heads ===
                _check_pos = [p for p in [5, 10, 20, 50, 100] if p < _S]
                for ki in range(K):
                    with torch.no_grad():
                        _logits = head(tgt[ki][0:1, :, :])  # [1, S, V]
                        _tok_ids = [_logits[0, p, :].argmax().item() for p in _check_pos]
                        _norms = [tgt[ki][0, p, :].norm().item() for p in _check_pos]
                    log_main(f"  [SHIFT] head{ki}: pos={_check_pos} -> tokens={_tok_ids}, norms=[{', '.join(f'{n:.1f}' for n in _norms)}]")
                # Check: are all heads identical? (should be False after bug fix)
                _all_same = True
                for p in _check_pos:
                    for ki in range(1, K):
                        if not torch.equal(tgt[0][0, p, :], tgt[ki][0, p, :]):
                            _all_same = False
                            break
                    if not _all_same:
                        break
                log_main(f"  [SHIFT] ALL HEADS SAME? => {_all_same} {'*** BUG! ***' if _all_same else '(correct: heads differ)'}")

            # --- Per-head loss computation (memory-efficient) ---
            # Instead of concat [K*B, S, V] logits, compute one head at a time
            total_vloss = torch.tensor(0.0, device=predict.device)
            total_ploss = torch.tensor(0.0, device=predict.device)
            _batch_correct = 0
            _batch_ct = 0
            _batch_topk = [torch.tensor(0.0, device=predict.device) for _ in range(3)]
            _per_head_vloss = []
            _per_head_ploss = []
            _per_head_acc = []

            for k in range(K):
                pred_k = predict[k]   # [B, S, H]
                tgt_k = tgt[k]        # [B, S, H]

                with torch.no_grad():
                    target_head_k = head(tgt_k)                        # [B, S, V]
                    target_p_k = nn.Softmax(dim=2)(target_head_k)      # [B, S, V]

                out_head_k = head(pred_k)                              # [B, S, V]
                out_logp_k = nn.LogSoftmax(dim=2)(out_head_k)          # [B, S, V]
                plogp = target_p_k * out_logp_k
                ploss_k = -torch.sum(torch.sum(loss_mask_base * plogp, 2)) / (loss_mask_base.sum() + 1e-5)

                vloss_k = criterion(pred_k, tgt_k)
                vloss_k = torch.sum(torch.mean(loss_mask_base * vloss_k, 2)) / (loss_mask_base.sum() + 1e-5)

                total_vloss = total_vloss + vloss_k
                total_ploss = total_ploss + ploss_k

                # Per-head accuracy (no grad)
                with torch.no_grad():
                    _, predicted_k = torch.max(out_head_k, 2)
                    _, target_k = torch.max(target_head_k, 2)
                    ct_k = loss_mask_base.sum().item()
                    cc_k = ((predicted_k == target_k) * loss_mask_base.squeeze(-1)).sum().item()
                    _batch_ct += ct_k
                    _batch_correct += cc_k
                    out_flat = out_head_k.reshape(-1, out_head_k.shape[-1])[loss_mask_base.reshape(-1) == 1]
                    tgt_flat = target_k.reshape(-1)[loss_mask_base.squeeze(-1).reshape(-1) == 1]
                    topkacc_k = top_accuracy(out_flat, tgt_flat, (1, 2, 3))
                    for ti in range(3):
                        _batch_topk[ti] = _batch_topk[ti] + topkacc_k[ti]

                _per_head_vloss.append(vloss_k.item())
                _per_head_ploss.append(ploss_k.item())
                _per_head_acc.append(cc_k / max(ct_k, 1))
                del target_head_k, target_p_k, out_head_k, out_logp_k, plogp

            # --- Per-head breakdown logging (first 5 batches + every 500) ---
            if accelerator.is_main_process and (batch_idx < 5 or batch_idx % 500 == 0):
                for ki in range(K):
                    log_main(f"  head{ki}: vloss={_per_head_vloss[ki]:.4f}, ploss={_per_head_ploss[ki]:.4f}, acc={_per_head_acc[ki]:.4f}")
                _mem_alloc = torch.cuda.memory_allocated() / 1024**3
                _mem_reserved = torch.cuda.memory_reserved() / 1024**3
                log_main(f"  GPU mem: {_mem_alloc:.2f}GB alloc, {_mem_reserved:.2f}GB reserved")
                # Verify per-head vloss differs (if heads learn differently, they should)
                if len(set(round(v, 6) for v in _per_head_vloss)) == 1:
                    log_main(f"  [WARN] All heads have identical vloss — may indicate a problem")

            vloss = total_vloss / K
            ploss = total_ploss / K
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss

            # --- Gradient flow check (first 5 batches) ---
            if accelerator.is_main_process and batch_idx < 5:
                log_main(f"  [GRAD] loss={loss.item():.6f}, loss.requires_grad={loss.requires_grad}, loss.grad_fn={type(loss.grad_fn).__name__}")

            accelerator.backward(loss)

            # --- Verify gradients actually reached model params (first 5 batches) ---
            if accelerator.is_main_process and batch_idx < 5:
                _has_grad = {}
                for pname, p in model.named_parameters():
                    if p.requires_grad:
                        _has_grad[pname] = p.grad is not None and p.grad.norm().item() > 0 if p.grad is not None else False
                log_main(f"  [GRAD] param grads: {_has_grad}")

            if accelerator.sync_gradients:
                grad_norm_before = accelerator.clip_grad_norm_(
                    model.parameters(), train_config["grad_clip"]
                )
                optimizer.step()
                if is_warmup:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Detailed per-optimizer-step logging (every 50 steps + first 5)
                if accelerator.is_main_process and (global_step <= 5 or global_step % 50 == 0):
                    cur_lr = optimizer.param_groups[0]['lr']
                    param_norms = {}
                    for pname, p in model.named_parameters():
                        if p.requires_grad:
                            param_norms[pname] = p.data.norm().item()
                    pnorm_str = ', '.join(f"{k}={v:.4f}" for k, v in param_norms.items())
                    log_main(
                        f"[step {global_step}] lr={cur_lr:.8f}, "
                        f"loss={loss.item():.6f}, vloss={vloss.item():.6f}, ploss={ploss.item():.6f}, "
                        f"grad_norm={grad_norm_before:.4f}, "
                        f"param_norms: {pnorm_str}"
                    )

        ct = _batch_ct
        cc = _batch_correct
        for top_i in range(3):
            top_3acc[top_i] += _batch_topk[top_i]
        total += ct
        correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/vloss": vloss.item(),
                "train/ploss": ploss.item(),
                "train/loss": loss.item(),
                "train/acc": cc / ct,
            }
            for id in range(3):
                logdict[f"train/top_{id + 1}_acc"] = _batch_topk[id].item() / ct
            wandb_log(logdict)

        epoch_loss += loss.item()
        epoch_vloss += vloss.item()
        epoch_ploss += ploss.item()
        num_batches += 1

        del ploss, vloss

    _epoch_elapsed = _time.time() - _epoch_start
    log_main(f"--- Epoch {epoch} TRAIN done in {_epoch_elapsed:.1f}s ({num_batches} batches, {global_step} total optimizer steps) ---")
    correct, total = torch.tensor(correct).to(accelerator.device), torch.tensor(total).to(accelerator.device)
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    epoch_vloss /= num_batches
    epoch_ploss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_main_process:
        for id, i in enumerate(top_3acc):
            wandb_log({f"train/epochtop_{id + 1}_acc": i.sum().item() / total})
    if accelerator.is_main_process:
        print(
            "Epoch [{}/{}], Loss: {:.4f}, Vloss: {:.4f}, Ploss: {:.4f}".format(
                epoch, num_epochs, epoch_loss, epoch_vloss, epoch_ploss
            )
        )
        print("Train Accuracy: {:.2f}%".format(100 * correct / total))
        wandb_log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    log_main(f"--- Epoch {epoch} TEST start ---")
    if True:  # eval every epoch, matching ViSpec_ping
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        epoch_vloss = 0
        epoch_ploss = 0
        num_batches = 0
        model.eval()

        for batch_idx, data in enumerate(
            tqdm(test_loader, disable=not accelerator.is_local_main_process)
        ):
            with torch.no_grad():
                hidden_states = data["hidden_states"]
                B = hidden_states.shape[0]
                predict_b = []
                for b in range(B):
                    pred_b = model(hidden_states[b : b + 1])
                    predict_b.append(pred_b)
                predict = torch.stack(predict_b, dim=1)  # [K, B, S, H]

                K = predict.shape[0]

                tgt = []
                for i in range(K):
                    shifted = torch.cat(
                        (
                            data["target"][:, i:],
                            torch.zeros_like(data["target"][:, :i]),
                        ),
                        dim=1,
                    )
                    tgt.append(shifted)

                loss_mask_base = data["loss_mask"][:, :, None]  # [B, S, 1]

                # === TEST SHIFT VERIFICATION (first 3 batches) ===
                if accelerator.is_main_process and batch_idx < 3:
                    _S = data["target"].shape[1]
                    _check_pos = [p for p in [5, 10, 20, 50, 100] if p < _S]
                    for ki in range(K):
                        with torch.no_grad():
                            _logits = head(tgt[ki][0:1, :, :])  # [1, S, V]
                            _tok_ids = [_logits[0, p, :].argmax().item() for p in _check_pos]
                        log_main(f"  [SHIFT] head{ki}: pos={_check_pos} -> tokens={_tok_ids}")
                    _all_same = all(
                        torch.equal(tgt[0][0, p, :], tgt[ki][0, p, :])
                        for p in _check_pos for ki in range(1, K)
                    )
                    log_main(f"  [SHIFT] ALL HEADS SAME? => {_all_same} {'*** BUG! ***' if _all_same else '(correct)'}")

                # Per-head loss & accuracy (memory-efficient)
                total_vloss = torch.tensor(0.0, device=predict.device)
                total_ploss = torch.tensor(0.0, device=predict.device)
                _batch_correct = 0
                _batch_ct = 0
                _batch_topk = [torch.tensor(0.0, device=predict.device) for _ in range(3)]
                _per_head_vloss = []
                _per_head_ploss = []
                _per_head_acc = []

                for k in range(K):
                    pred_k = predict[k]   # [B, S, H]
                    tgt_k = tgt[k]        # [B, S, H]

                    target_head_k = head(tgt_k)                        # [B, S, V]
                    target_p_k = nn.Softmax(dim=2)(target_head_k)      # [B, S, V]

                    out_head_k = head(pred_k)                          # [B, S, V]
                    out_logp_k = nn.LogSoftmax(dim=2)(out_head_k)
                    plogp = target_p_k * out_logp_k
                    ploss_k = -torch.sum(torch.sum(loss_mask_base * plogp, 2)) / (loss_mask_base.sum() + 1e-5)

                    vloss_k = criterion(pred_k, tgt_k)
                    vloss_k = torch.sum(torch.mean(loss_mask_base * vloss_k, 2)) / (loss_mask_base.sum() + 1e-5)

                    total_vloss = total_vloss + vloss_k
                    total_ploss = total_ploss + ploss_k

                    _, predicted_k = torch.max(out_head_k, 2)
                    _, target_k = torch.max(target_head_k, 2)
                    ct_k = loss_mask_base.sum().item()
                    cc_k = ((predicted_k == target_k) * loss_mask_base.squeeze(-1)).sum().item()
                    _batch_ct += ct_k
                    _batch_correct += cc_k
                    out_flat = out_head_k.reshape(-1, out_head_k.shape[-1])[loss_mask_base.reshape(-1) == 1]
                    tgt_flat = target_k.reshape(-1)[loss_mask_base.squeeze(-1).reshape(-1) == 1]
                    topkacc_k = top_accuracy(out_flat, tgt_flat, (1, 2, 3))
                    for ti in range(3):
                        _batch_topk[ti] = _batch_topk[ti] + topkacc_k[ti]

                    _per_head_vloss.append(vloss_k.item())
                    _per_head_ploss.append(ploss_k.item())
                    _per_head_acc.append(cc_k / max(ct_k, 1))
                    del target_head_k, target_p_k, out_head_k, out_logp_k, plogp

                # Log first 3 test batches per-head breakdown
                if accelerator.is_main_process and batch_idx < 3:
                    _S = data["target"].shape[1]
                    log_main(f"[TEST  batch={batch_idx}] B={B}, S={_S}, K={K}")
                    for ki in range(K):
                        log_main(f"  head{ki}: vloss={_per_head_vloss[ki]:.4f}, ploss={_per_head_ploss[ki]:.4f}, acc={_per_head_acc[ki]:.4f}")

                vloss = total_vloss / K
                ploss = total_ploss / K
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                for top_i in range(3):
                    top_3acc[top_i] += _batch_topk[top_i]
                total += _batch_ct
                correct += _batch_correct
            epoch_loss += loss.item()
            epoch_vloss += vloss.item()
            epoch_ploss += ploss.item()
            num_batches += 1

        correct, total = torch.tensor(correct).to(accelerator.device), torch.tensor(total).to(accelerator.device)
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_main_process:
            for id, i in enumerate(top_3acc):
                wandb_log({f"test/top_{id + 1}_acc": i.sum().item() / total})
        epoch_loss /= num_batches
        epoch_vloss /= num_batches
        epoch_ploss /= num_batches
        if accelerator.is_main_process:
            print(
                "Test Epoch [{}/{}], Loss: {:.4f}, Vloss: {:.4f}, Ploss: {:.4f}".format(
                    epoch, num_epochs, epoch_loss, epoch_ploss, epoch_vloss
                )
            )
            print("Test Accuracy: {:.2f}%".format(100 * correct / total))
            wandb_log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})

        # save_state is a collective op (FSDP gather) — ALL ranks must call it
        save_dir = f"{args.cpdir}/state_{epoch}"
        log_main(f"Saving checkpoint to {save_dir} ...")
        accelerator.save_state(output_dir=save_dir)

        # Only rank 0 writes extra metadata files
        if accelerator.is_main_process:
            import shutil
            shutil.copyfile(args.configpath, f"{save_dir}/config.json")
            # Save resume metadata for auto-resume
            resume_meta = {"epoch": epoch, "global_step": global_step}
            with open(os.path.join(save_dir, "resume_meta.json"), "w") as f:
                json.dump(resume_meta, f)
            # Log saved files
            if os.path.exists(save_dir):
                saved_files = os.listdir(save_dir)
                total_size = sum(os.path.getsize(os.path.join(save_dir, f)) for f in saved_files if os.path.isfile(os.path.join(save_dir, f)))
                log_main(f"Checkpoint saved: {len(saved_files)} files, {total_size/1024/1024:.1f} MB")
                for sf in sorted(saved_files):
                    fpath = os.path.join(save_dir, sf)
                    if os.path.isfile(fpath):
                        log_main(f"  {sf}: {os.path.getsize(fpath)/1024:.1f} KB")
            log_main(f"--- Epoch {epoch} DONE ---")
            log_main("="*70)
