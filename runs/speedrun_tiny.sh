#!/bin/bash

# This script is a "tiny" version of speedrun.sh, designed for fast iteration on 2x RTX 2000 Ada (16GB) GPUs.
# It runs the full pipeline (Data -> Train -> SFT) but with a much smaller model and dataset subset.


SECONDS=0

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# NCCL configuration for distributed training
# If you encounter P2P issues, set NCCL_P2P_DISABLE=1
export NCCL_P2P_DISABLE=1


# Usage: bash runs/speedrun_tiny.sh [stage]
# Stages:
#   all   : Run everything (default)
#   setup : Install dependencies, download data, train tokenizer
#   train : Run pretraining only
#   sft   : Run SFT (Supervised Fine-Tuning) only
#   rl    : Run RL (reinforcement learning on GSM8K) only
#
# Examples:
#   bash runs/speedrun_tiny.sh        # Runs full pipeline
#   bash runs/speedrun_tiny.sh train  # Runs only pretraining
#   bash runs/speedrun_tiny.sh rl     # Runs only RL (requires SFT checkpoints)

STAGE=${1:-all}

echo "Running stage: $STAGE"

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=tiny-run bash runs/speedrun_tiny.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Setup & Tokenizer (Runs for 'all', 'setup', and always checks env)
SECTION_START=$SECONDS
if [[ "$STAGE" == "all" || "$STAGE" == "setup" ]]; then
    # Python venv setup with uv
    if [ ! -d ".venv" ]; then
        command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
        uv venv
        uv sync --extra gpu
    fi
    source .venv/bin/activate

    # Report reset
    if [ "$STAGE" == "all" ]; then
        python -m nanochat.report reset
    fi

    # Tokenizer
    # Only prepare data/tokenizer if tokenizer doesn't exist
    if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer.model" ]; then
        # Download just enough data for a tiny run (8 shards = ~2B chars).
        python -m nanochat.dataset -n 8

        # train the tokenizer
        python -m scripts.tok_train
        # evaluate the tokenizer
        python -m scripts.tok_eval
    else
        echo "Tokenizer already exists."
    fi
    SECTION_DURATION=$(($SECONDS - $SECTION_START))
    echo "Setup & Tokenizer completed in $(($SECTION_DURATION / 60)) minutes and $(($SECTION_DURATION % 60)) seconds."
else
    # Always activate venv for other stages
    source .venv/bin/activate
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    SECTION_START=$SECONDS
    echo "Starting Pretraining..."
    # Train with adjusted settings:
    # - No --fp8 (avoid NaN on tiny models)
    # - --core-metric-every=-1 (avoid crashing on long-context tasks like SQuAD)
    # - --save-every=50 (ensure we have checkpoints even if it crashes/stops early)
    torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
        --depth=4 \
        --max-seq-len=256 \
        --target-param-data-ratio=5.0 \
        --device-batch-size=32 \
        --core-metric-every=-1 \
        --save-every=50 \
        --run=$WANDB_RUN

    # evaluate the model (skip core metrics to avoid Sequence length errors)
    torchrun --standalone --nproc_per_node=2 -m scripts.base_eval -- \
        --device-batch-size=32 \
        --eval bpb,sample
    SECTION_DURATION=$(($SECONDS - $SECTION_START))
    echo "Pretraining completed in $(($SECTION_DURATION / 60)) minutes and $(($SECTION_DURATION % 60)) seconds."
fi

# -----------------------------------------------------------------------------
# SFT
if [[ "$STAGE" == "all" || "$STAGE" == "sft" ]]; then
    SECTION_START=$SECONDS
    echo "Starting SFT..."
    # download identity conversations
    curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

    # run SFT and eval
    torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- \
        --device-batch-size=32 \
        --run=$WANDB_RUN

    # Run only lightweight evaluation (SpellingBee) to avoid timeout/OOM/Sequence length issues
    torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i sft -a SpellingBee
    SECTION_DURATION=$(($SECONDS - $SECTION_START))
    echo "SFT completed in $(($SECTION_DURATION / 60)) minutes and $(($SECTION_DURATION % 60)) seconds."
fi

# -----------------------------------------------------------------------------
# RL (reinforcement learning on GSM8K; requires SFT checkpoints)
if [[ "$STAGE" == "rl" ]]; then
    SECTION_START=$SECONDS
    source .venv/bin/activate
    echo "Starting RL..."
    torchrun --standalone --nproc_per_node=2 -m scripts.chat_rl -- \
        --device-batch-size=4 \
        --run=$WANDB_RUN
    SECTION_DURATION=$(($SECONDS - $SECTION_START))
    echo "RL completed in $(($SECTION_DURATION / 60)) minutes and $(($SECTION_DURATION % 60)) seconds."
fi

# -----------------------------------------------------------------------------
# Generate report
if [ "$STAGE" == "all" ]; then
    SECTION_START=$SECONDS
    python -m nanochat.report generate
    echo "Speedrun Tiny Complete. Report generated."
    SECTION_DURATION=$(($SECONDS - $SECTION_START))
    echo "Report generation completed in $(($SECTION_DURATION / 60)) minutes and $(($SECTION_DURATION % 60)) seconds."
fi

duration=$SECONDS
echo "Total execution time for stage '$STAGE': $(($duration / 60)) minutes and $(($duration % 60)) seconds."
