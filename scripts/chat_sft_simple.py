"""
Simplified educational SFT script (staged).

Purpose: Learn SFT in stages—Stage 1 single-GPU skeleton, Stage 2 DDP/multi-GPU,
Stage 3 BOS-aligned best-fit dataloader, Stage 4 task mixture. Metrics (loss, lrm, dt,
tok/sec, mfu) match scripts/chat_sft.py for direct comparison.

How to run:
  Use the project .venv:  source .venv/bin/activate  (or run via  uv run).

  Stage 1 (single-GPU, no torchrun):
    source .venv/bin/activate
    python -m scripts.chat_sft_simple --stage 1 --device-batch-size=32 --run=dummy

  Stage 2+ (multi-GPU, use torchrun):
    source .venv/bin/activate
    torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft_simple -- --stage 2 --device-batch-size=32 --run=dummy
    (Use nproc_per_node=N for N GPUs; same as runs/speedrun_tiny.sh SFT invocation.)

Prerequisite: A pre-trained base model must exist (e.g. from pretraining or from
  runs/speedrun_tiny.sh). If the script cannot find a base checkpoint, it will
  print instructions and exit. Create a base model first, e.g.:
    bash runs/speedrun_tiny.sh train
  or
    bash runs/speedrun_tiny.sh all


torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft_simple -- --stage 1 --device-batch-size=32 --run=dummy --num-iterations=10 --eval-every=10

## stage 1:

step 00010 (100.00%) | loss: 4.379523 | lrm: 0.00 | dt: 426.75ms | tok/sec: 153,568 | mfu: 1.11 | epoch: 1 | total time: 0.00m
  [per-piece ms] data: 8.2 | fwd: 70.2 | bwd: 108.2 | opt: 41.1
Peak memory usage: 7027.53 MiB
Total training time: 0.00m
Minimum validation bpb: 1.4774
Per-piece (avg ms): data 8.2 | fwd 70.2 | bwd 108.2 | opt 41.1

## stage 2:

step 00010 (100.00%) | loss: 4.051101 | lrm: 0.00 | dt: 1637.55ms | tok/sec: 320,166 | mfu: 1.15 | epoch: 1 | total time: 0.00m
  [per-piece ms] data: 17.2 | fwd: 105.8 | bwd: 233.7 | opt: 44.2
Step 00010 | Validation bpb: 1.3456 | eval time: 10997.4ms
Peak memory usage: 7119.54 MiB
Total training time: 0.00m
Minimum validation bpb: 1.3456
Per-piece (avg ms): data 17.2 | fwd 105.8 | bwd 233.7 | opt 44.2

## stage 3:
step 00010 (100.00%) | loss: 4.018104 | lrm: 0.00 | dt: 1736.40ms | tok/sec: 301,939 | mfu: 1.09 | epoch: 1 | total time: 0.00m
  [per-piece ms] data: 33.3 | fwd: 108.1 | bwd: 241.5 | opt: 41.3
Peak memory usage: 7119.54 MiB
Total training time: 0.00m
Minimum validation bpb: 1.3077
Per-piece (avg ms): data 33.3 | fwd 108.1 | bwd 241.5 | opt 41.3





"""

import argparse
import os
import sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from contextlib import nullcontext
from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    get_base_dir,
    autodetect_device_type,
)
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import (
    save_checkpoint,
    load_model,
    find_largest_model,
    find_last_step,
)
from nanochat.loss_eval import evaluate_bpb
import torch.distributed as dist

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# 1) CLI arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Simplified educational SFT (staged: 1=single-GPU, 2=DDP, 3=BOS dataloader, 4=task mix)"
)
parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
    help="Stage 1: single-GPU. Stage 2: DDP. Stage 3: BOS best-fit dataloader. Stage 4: task mixture.")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' = no wandb)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--num-iterations", type=int, default=-1, help="optimization steps (-1 = full epoch)")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens (Stage 1 overrides to device_batch_size * max_seq_len)")
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--init-lr-frac", type=float, default=1.0)
parser.add_argument("--eval-every", type=int, default=150, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="tokens for val eval")
parser.add_argument("--dry-run", action="store_true", help="skip checkpoints/report")
parser.add_argument("--no-compile", action="store_true", help="disable torch.compile for readability")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# 2) Pre-trained model check: require base checkpoint before starting
# -----------------------------------------------------------------------------
def _check_base_model_exists():
    """Ensure a base (pre-trained) checkpoint exists. If not, print instructions and exit."""
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    if not os.path.isdir(checkpoints_dir):
        print("No base checkpoints directory found.", file=sys.stderr)
        print(f"  Expected: {checkpoints_dir}", file=sys.stderr)
        print("  Create a base model first by running:", file=sys.stderr)
        print("    bash runs/speedrun_tiny.sh train", file=sys.stderr)
        print("  or:", file=sys.stderr)
        print("    bash runs/speedrun_tiny.sh all", file=sys.stderr)
        sys.exit(1)
    try:
        model_tag = args.model_tag or find_largest_model(checkpoints_dir)
        checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
        find_last_step(checkpoint_dir)
    except FileNotFoundError as e:
        print("Base checkpoint not found.", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print("  Create a base model first by running:", file=sys.stderr)
        print("    bash runs/speedrun_tiny.sh train", file=sys.stderr)
        print("  or: bash runs/speedrun_tiny.sh all", file=sys.stderr)
        sys.exit(1)

_check_base_model_exists()

# -----------------------------------------------------------------------------
# 3) Compute init: Stage 1 = single process; Stage 2+ = DDP when run with torchrun
# -----------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# Stage 1: treat as single-GPU (no grad accum, total_batch = one step)
if args.stage == 1:
    ddp_world_size = 1
    args.total_batch_size = args.device_batch_size * args.max_seq_len

master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

if args.stage >= 2 and ddp_world_size == 1:
    print0("Note: Stage 2+ is intended for multi-GPU. Run with: torchrun --standalone --nproc_per_node=N -m scripts.chat_sft_simple -- ...")

# -----------------------------------------------------------------------------
# 4) Wandb, model load, optimizer, LR/momentum schedules
# -----------------------------------------------------------------------------
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and args.device_batch_size > pretrain_batch_size:
    print0(f"FOOTGUN WARNING: base model used device_batch_size {pretrain_batch_size}; you passed --device-batch-size={args.device_batch_size}")
orig_model = model
if not args.no_compile:
    model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Stage: {args.stage} | Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch (world): {world_tokens_per_fwdbwd:,} | Total batch {args.total_batch_size:,} => grad_accum_steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

def get_lr_multiplier(progress):
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# -----------------------------------------------------------------------------
# 5) Datasets and data generators
# -----------------------------------------------------------------------------
base_dir = get_base_dir()

# Stage 1 & 2: single task (SmolTalk). Stage 3: SmolTalk only. Stage 4: mixture.
if args.stage <= 3:
    train_dataset = SmolTalk(split="train")
    val_dataset = SmolTalk(split="test")
else:
    # Stage 4: small task mixture (aligned with chat_sft.py structure, reduced)
    train_dataset = TaskMixture([
        SmolTalk(split="train"),
        MMLU(subset="auxiliary_train", split="train"),
        GSM8K(subset="main", split="train"),
    ])
    val_dataset = TaskMixture([
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ])

# Globals updated by the data generator (for last_step, progress, epoch)
last_step = False
approx_progress = 0.0
current_epoch = 1


def simple_dataloader(split, max_seq_len, device_batch_size, row_capacity, bos_token, dataset, tokenizer, device, device_type, ddp_rank, ddp_world_size):
    """
    Simple one-conversation-per-row dataloader (Stage 1 & 2).
    Each row is one conversation: tokenize, truncate to max_seq_len+1, pad with BOS, mask padding in targets to -1.

    Multi-GPU note: Each rank builds its own full batch (device_batch_size rows); we do not split
    one batch across ranks. So with 2 GPUs you have 2 processes each doing the same per-batch work
    (tokenization, tensor build). They often share one machine and contend for CPU/memory, so
    per-rank data time can be higher in Stage 2 than in Stage 1—not lower. To speed up data loading
    when scaling, use more DataLoader workers per rank or pre-tokenized/sharded data.
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}
    ds = dataset
    size = len(ds)
    cursor = ddp_rank
    epoch = 1
    it = 0  # number of batch yields so far
    use_cuda = device_type == "cuda"
    # num_iterations is in optimizer steps; each step consumes grad_accum_steps batches
    grad_accum_steps = args.total_batch_size // (args.device_batch_size * args.max_seq_len * ddp_world_size)

    while True:
        rows = []
        row_lengths = []
        for _ in range(device_batch_size):
            conv = ds[cursor % size]
            cursor += ddp_world_size
            if cursor >= size:
                cursor = cursor % size
                epoch += 1
            ids, _ = tokenizer.render_conversation(conv, max_tokens=row_capacity)
            content_len = min(len(ids), row_capacity)
            row = ids[:row_capacity]
            if len(row) < row_capacity:
                row = row + [bos_token] * (row_capacity - len(row))
            rows.append(row)
            row_lengths.append(content_len)

        it += 1
        if split == "train":
            current_epoch = epoch
            # num_iterations = optimizer steps; batches per step = grad_accum_steps
            if args.num_iterations > 0:
                steps_so_far = (it - 1) // grad_accum_steps  # completed full steps worth of batches
                approx_progress = min(1.0, steps_so_far / args.num_iterations)
                # stop after yielding enough batches for num_iterations optimizer steps (incl. prefetch)
                if it >= 1 + args.num_iterations * grad_accum_steps:
                    last_step = True
            else:
                approx_progress = (it * device_batch_size * ddp_world_size) / max(size, 1)
            if it * device_batch_size * ddp_world_size >= size:
                last_step = True

        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1 :] = -1

        yield inputs, targets


def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader with best-fit packing (Stage 3 & 4).

    - BOS-aligned: every row starts with BOS so the model always sees conversation boundaries.
    - Best-fit: pick the largest conversation that fits in the remaining space to minimize padding.
    - Pad (don't crop): when no conversation fits, pad the row with BOS and set targets to -1 for padding.
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()

    conv_buffer = []
    cursor = ddp_rank
    consumed = ddp_rank
    epoch = 1
    it = 0  # number of batch yields so far
    grad_accum_steps = args.total_batch_size // (args.device_batch_size * args.max_seq_len * ddp_world_size)

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conv = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conv)
            conv_buffer.append(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows = []
        row_lengths = []
        for _ in range(args.device_batch_size):
            row = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, conv in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len
                if best_idx >= 0:
                    conv = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    consumed += ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    padded = True
                    break
            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])

        it += 1
        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                steps_so_far = (it - 1) // grad_accum_steps
                approx_progress = min(1.0, steps_so_far / args.num_iterations)
                if it >= 1 + args.num_iterations * grad_accum_steps:
                    last_step = True
            else:
                approx_progress = consumed / dataset_size
            if consumed >= dataset_size:
                last_step = True

        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1 :] = -1
        yield inputs, targets


row_capacity = args.max_seq_len + 1
bos_token = tokenizer.get_bos_token_id()

if args.stage <= 2:
    def train_data_gen():
        return simple_dataloader(
            "train", args.max_seq_len, args.device_batch_size, row_capacity, bos_token,
            train_dataset, tokenizer, device, device_type, ddp_rank, ddp_world_size,
        )
    def build_val_loader():
        return simple_dataloader(
            "val", args.max_seq_len, args.device_batch_size, row_capacity, bos_token,
            val_dataset, tokenizer, device, device_type, ddp_rank, ddp_world_size,
        )
else:
    train_data_gen = lambda: sft_data_generator_bos_bestfit("train")
    build_val_loader = lambda: sft_data_generator_bos_bestfit("val")

train_loader = train_data_gen()
progress = 0

# -----------------------------------------------------------------------------
# 6) Training loop: same structure as chat_sft.py, with per-piece timing
# -----------------------------------------------------------------------------
x, y = next(train_loader)
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0

# Per-piece timing (running averages in ms)
avg_dt_data = 0.0
avg_dt_fwd = 0.0
avg_dt_bwd = 0.0
avg_dt_opt = 0.0
ema_piece = 0.95

while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        t_eval0 = time.time()
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        model.train()
        synchronize()
        dt_eval = (time.time() - t_eval0) * 1000
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f} | eval time: {dt_eval:.1f}ms")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })

    if master_process and last_step and not args.dry_run:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,
            },
        )

    if last_step:
        break

    # --- Single training step with per-piece timing ---
    synchronize()
    t0 = time.time()

    t_data0 = time.time()
    for micro_step in range(grad_accum_steps):
        t_fwd0 = time.time()
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        synchronize()
        dt_fwd = (time.time() - t_fwd0) * 1000

        t_bwd0 = time.time()
        loss.backward()
        synchronize()
        dt_bwd = (time.time() - t_bwd0) * 1000

        t_data1 = time.time()
        x, y = next(train_loader)
        dt_data_ms = (time.time() - t_data1) * 1000
        progress = max(progress, approx_progress)

        avg_dt_data = ema_piece * avg_dt_data + (1 - ema_piece) * dt_data_ms
        avg_dt_fwd = ema_piece * avg_dt_fwd + (1 - ema_piece) * dt_fwd
        avg_dt_bwd = ema_piece * avg_dt_bwd + (1 - ema_piece) * dt_bwd

    t_opt0 = time.time()
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group.get("kind") == "muon":
            group["momentum"] = muon_momentum
    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    dt_opt = (time.time() - t_opt0) * 1000
    avg_dt_opt = ema_piece * avg_dt_opt + (1 - ema_piece) * dt_opt

    t1 = time.time()
    dt = t1 - t0
    step += 1

    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    if step > 10:
        total_training_time += dt

    # Main step line (identical format to chat_sft.py for comparison)
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")
    # Per-piece timing (extra line for learning)
    print0(f"  [per-piece ms] data: {avg_dt_data:.1f} | fwd: {avg_dt_fwd:.1f} | bwd: {avg_dt_bwd:.1f} | opt: {avg_dt_opt:.1f}")

    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })

# -----------------------------------------------------------------------------
# 7) Final stats: per-piece summary, peak memory, total time, min val bpb
# -----------------------------------------------------------------------------
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")
print0(f"Per-piece (avg ms): data {avg_dt_data:.1f} | fwd {avg_dt_fwd:.1f} | bwd {avg_dt_bwd:.1f} | opt {avg_dt_opt:.1f}")

if not args.dry_run:
    from nanochat.report import get_report
    get_report().log(section="SFT", data=[
        user_config,
        {"Number of iterations": step, "DDP world size": ddp_world_size, "Stage": args.stage},
        {"Minimum validation bpb": min_val_bpb},
    ])

wandb_run.finish()
compute_cleanup()
