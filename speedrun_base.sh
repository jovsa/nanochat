#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# Disable P2P communication between GPUs to avoid conflicts for  RTX 4000 Ada GPUs
export NCCL_P2P_DISABLE=1

mkdir -p $NANOCHAT_BASE_DIR
# -----------------------------------------------------------------------------
# Python venv setup with uv

# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)

# Number of processes/GPUs to use
NPROC_PER_NODE=2

# pretrain the d20 model
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=10 --device_batch_size=8 --num_iterations=10 --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20 --run=$WANDB_RUN

# version 2
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=4 --max_seq_len=512 --device_batch_size=2 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=2048 --num_iterations=20 --run=$WANDB_RUN

# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss --device_batch_size=2
# evaluate the model on CORE tasks
# NOTE: we set the max-per-task to 20 here to match the eval bundle. It cannot be less than 20.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --max-per-task=20
