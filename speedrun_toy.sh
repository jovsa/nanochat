#!/bin/bash

# TOY VERSION of speedrun.sh for quick iteration (~10 minutes)
# This script runs the full pipeline with drastically reduced parameters.
# Useful for testing changes to the codebase without waiting for the full run.

# Key differences from speedrun.sh:
# - Dataset: 2 shards instead of 240
# - Tokenizer: 50M chars instead of 2B
# - Model: depth=4 (~4.5M params) instead of depth=10 (~56M params)
# - Sequence length: 512 instead of 2048
# - Batch size: 8192 instead of 524288
# - Iterations: 5 instead of 10 for each stage
# - RL: included (commented out in original), with reduced samples/evals

# 1) Example launch (simplest):
# bash speedrun_toy.sh
# 2) Example launch in a screen session:
# screen -L -Logfile speedrun_toy.log -S speedrun_toy bash speedrun_toy.sh

# Start timing
START_TIME=$SECONDS

# Helper function to print elapsed time
print_elapsed() {
    local elapsed=$((SECONDS - START_TIME))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    echo "⏱️  [$1] Elapsed time: ${mins}m ${secs}s"
}

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# Disable P2P communication between GPUs to avoid conflicts for RTX 4000 Ada GPUs
export NCCL_P2P_DISABLE=1

# Number of processes/GPUs to use
NPROC_PER_NODE=2

mkdir -p $NANOCHAT_BASE_DIR
# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# Dataset and Tokenizer Training (TOY: reduced data)

# Download only 2 shards (~500M chars, ~200MB on disk)
python -m nanochat.dataset -n 2

# Train the tokenizer on 50M characters (instead of 2B)
python -m scripts.tok_train --max_chars=50000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval
print_elapsed "Tokenizer"

# -----------------------------------------------------------------------------
# Base model (pretraining) - TOY VERSION

# The depth=4 model is ~4.5M parameters (vs 561M for depth=20)
# With reduced batch size and iterations, this trains in ~2-3 minutes

# pretrain the tiny d4 model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=2 \
    --total_batch_size=8192 \
    --eval_tokens=8192 \
    --core_metric_every=-1 \
    --num_iterations=5 \
    --run=$WANDB_RUN

# evaluate the model on a small chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss --device_batch_size=2
# evaluate the model on CORE tasks (minimal set)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --max-per-task=20
print_elapsed "Base model"

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining with reduced parameters
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- \
    --device_batch_size=2 \
    --total_batch_size=8192 \
    --eval_tokens=8192 \
    --num_iterations=5 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid -a "MMLU|ARC-Easy"
print_elapsed "Midtraining"

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft with reduced iterations
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
    --device_batch_size=2 \
    --num_iterations=5 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft -a "MMLU|ARC-Easy"
print_elapsed "SFT"

# -----------------------------------------------------------------------------
# Reinforcement Learning on GSM8K (GRPO-style)

# run reinforcement learning with reduced parameters
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- \
    --device_batch_size=2 \
    --examples_per_step=4 \
    --num_samples=4 \
    --eval_every=10 \
    --save_every=20 \
    --eval_examples=50 \
    --run=$WANDB_RUN
# eval the RL model on GSM8K
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K
print_elapsed "RL"

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate

# Final timing
TOTAL_TIME=$((SECONDS - START_TIME))
TOTAL_MINS=$((TOTAL_TIME / 60))
TOTAL_SECS=$((TOTAL_TIME % 60))

echo "========================================"
echo "TOY SPEEDRUN COMPLETE!"
echo "Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "========================================"

