# nanochat Codebase Mindmap

A comprehensive visual guide to the nanochat codebase structure, organized hierarchically.

## 🎯 Overview

nanochat is a full-stack implementation of a ChatGPT-like LLM, designed to train end-to-end on a single 8XH100 node for ~$100. This mindmap breaks down all major components and their relationships.

---

## 📦 Core Architecture

### GPT Model (`nanochat/gpt.py`)
- **Main Class**: `GPT(nn.Module)`
- **Key Components**:
  - `GPTConfig`: Model configuration (sequence_len, vocab_size, n_layer, n_head, n_kv_head, n_embd)
  - `Block`: Transformer block with attention + MLP
  - `CausalSelfAttention`: Multi-head attention with GQA support
  - `MLP`: Feed-forward network with ReLU² activation
- **Key Features**:
  - Rotary Position Embeddings (RoPE) - `_precompute_rotary_embeddings()`
  - QK Norm (normalizes queries and keys)
  - Group Query Attention (GQA) for efficient inference
  - RMSNorm (no learnable parameters)
  - Untied weights for token embedding and lm_head
  - No bias in linear layers
- **Key Methods**:
  - `forward()`: Main forward pass with optional KV cache
  - `generate()`: Naive autoregressive generation
  - `setup_optimizers()`: Creates Muon + AdamW optimizers
  - `estimate_flops()`: Calculate FLOPs per token

### Inference Engine (`nanochat/engine.py`)
- **Main Class**: `Engine`
- **Key Components**:
  - `KVCache`: Efficient KV cache for autoregressive generation
  - `RowState`: Per-row state tracking during generation
  - `sample_next_token()`: Token sampling with temperature/top_k
  - `use_calculator()`: Safe Python expression evaluation for tool use
- **Key Features**:
  - KV cache management with dynamic growth
  - Batch generation with prefill + decode
  - Tool use support (Python calculator)
  - Special token handling (`<|python_start|>`, `<|assistant_end|>`, etc.)
- **Key Methods**:
  - `generate()`: Streaming generation with KV cache
  - `generate_batch()`: Non-streaming batch generation

### Tokenizer (`nanochat/tokenizer.py`)
- **Main Classes**: `RustBPETokenizer`, `HuggingFaceTokenizer`
- **Key Features**:
  - GPT-4 style BPE tokenizer
  - Special tokens for conversations (`<|user_start|>`, `<|assistant_start|>`, etc.)
  - Rust implementation for training (rustbpe)
  - tiktoken for efficient inference
- **Key Methods**:
  - `encode()`: Tokenize text (string or list)
  - `decode()`: Detokenize token IDs
  - `render_conversation()`: Convert conversation dict to token IDs + mask
  - `render_for_completion()`: Prepare conversation for completion (RL)
- **Special Tokens**:
  - `<|bos|>`: Beginning of sequence
  - `<|user_start|>`, `<|user_end|>`: User message delimiters
  - `<|assistant_start|>`, `<|assistant_end|>`: Assistant message delimiters
  - `<|python_start|>`, `<|python_end|>`: Python tool call delimiters
  - `<|output_start|>`, `<|output_end|>`: Tool output delimiters

---

## 🚂 Training Pipeline

### Base Training (`scripts/base_train.py`)
- **Purpose**: Pretrain model on raw text data
- **Key Features**:
  - Distributed training with DDP
  - Gradient accumulation
  - Mixed precision (bfloat16)
  - Learning rate scheduling (warmup/warmdown)
  - Checkpoint saving/resuming
- **Key Hyperparameters**:
  - `depth`: Model depth (layers)
  - `device_batch_size`: Per-device batch size
  - `total_batch_size`: Total batch size in tokens
  - `num_iterations`: Training steps
  - `target_param_data_ratio`: Chinchilla ratio (default 20)
- **Evaluation**:
  - Validation bits per byte (bpb)
  - CORE metric evaluation
  - Model sampling

### Midtraining (`scripts/mid_train.py`)
- **Purpose**: Teach model conversation format, tool use, multiple choice
- **Key Features**:
  - Trains on task mixture (SmolTalk, MMLU, GSM8K, identity conversations, spelling)
  - Conversation rendering with masks
  - Lower learning rate than base training
- **Task Mixture**:
  - SmolTalk: General conversations
  - MMLU: Multiple choice questions
  - GSM8K: Math problems with calculator tool use
  - CustomJSON: Identity conversations
  - SimpleSpelling/SpellingBee: Spelling tasks

### Supervised Fine-tuning (`scripts/chat_sft.py`)
- **Purpose**: Domain adaptation to specific tasks
- **Key Features**:
  - Trains on smaller, curated task mixture
  - Variable-length sequences with padding
  - Mask-based loss (only supervise assistant tokens)
- **Task Mixture**:
  - ARC-Easy, ARC-Challenge
  - GSM8K
  - SmolTalk (subset)
  - Identity conversations
  - Spelling tasks

### Reinforcement Learning (`scripts/chat_rl.py`)
- **Purpose**: Optional RL fine-tuning (GSM8K focused)
- **Status**: Experimental/optional

---

## ⚙️ Optimizers

### Muon Optimizer (`nanochat/muon.py`)
- **Purpose**: Optimize 2D matrix parameters (linear layers)
- **Key Features**:
  - SGD-momentum with Nesterov
  - Newton-Schulz orthogonalization
  - Aspect-ratio scaled learning rate
  - Distributed version (`DistMuon`) with reduce-scatter/all-gather
- **Key Methods**:
  - `zeropower_via_newtonschulz5()`: Newton-Schulz iteration
  - `step()`: Optimizer step with orthogonalization
- **Usage**: Applied to transformer blocks (attention + MLP)

### AdamW Optimizer (`nanochat/adamw.py`)
- **Purpose**: Optimize embeddings and lm_head
- **Key Features**:
  - Distributed AdamW (ZeRO-2 style)
  - Sharded optimizer states
  - Gradient reduction via reduce-scatter
  - Weight synchronization via all-gather
- **Usage**: Applied to `wte` (token embeddings) and `lm_head`

---

## 📊 Data Management

### Dataset (`nanochat/dataset.py`)
- **Purpose**: Download and manage pretraining data
- **Key Features**:
  - Downloads FineWeb-Edu 100BT dataset shards
  - Parquet file format
  - Parallel downloads with retries
- **Key Functions**:
  - `list_parquet_files()`: List available parquet files
  - `parquets_iter_batched()`: Iterate over dataset batches
  - `download_single_file()`: Download with retry logic
- **Data Format**: Parquet files with 'text' column

### Dataloader (`nanochat/dataloader.py`)
- **Purpose**: Stream and tokenize pretraining data
- **Key Features**:
  - Distributed data loading (DDP-aware)
  - Tokenization on-the-fly
  - Approximate resume support
  - Memory pinning for CUDA
- **Key Functions**:
  - `tokenizing_distributed_data_loader_with_state()`: Main data loader with state
  - `tokenizing_distributed_data_loader()`: Simplified version without state
- **Data Flow**: Parquet → Row Groups → Text → Tokens → Batches

### Tasks (`tasks/`)
- **Base Class**: `Task` (`tasks/common.py`)
- **Task Types**:
  - `TaskMixture`: Mix multiple tasks for training
  - `TaskSequence`: Sequential task training (curriculum)
- **Available Tasks**:
  - `MMLU` (`tasks/mmlu.py`): Multiple choice across many subjects
  - `ARC` (`tasks/arc.py`): Science questions (Easy/Challenge)
  - `GSM8K` (`tasks/gsm8k.py`): Grade school math problems
  - `HumanEval` (`tasks/humaneval.py`): Python coding tasks
  - `SmolTalk` (`tasks/smoltalk.py`): General conversations
  - `SpellingBee` (`tasks/spellingbee.py`): Spelling/counting tasks
  - `CustomJSON` (`tasks/customjson.py`): Custom conversation datasets

---

## 🏗️ Infrastructure

### Checkpoint Manager (`nanochat/checkpoint_manager.py`)
- **Purpose**: Save/load model checkpoints
- **Key Functions**:
  - `save_checkpoint()`: Save model, optimizer, metadata
  - `load_checkpoint()`: Load checkpoint from disk
  - `build_model()`: Reconstruct model from checkpoint
  - `load_model()`: Convenience function for loading
  - `find_largest_model()`: Auto-detect model tag
  - `find_last_step()`: Auto-detect latest step
- **Checkpoint Structure**:
  - `model_{step:06d}.pt`: Model state dict
  - `optim_{step:06d}_rank{rank}.pt`: Optimizer state (per rank)
  - `meta_{step:06d}.json`: Training metadata

### Distributed Training (`nanochat/common.py`)
- **Purpose**: DDP setup and utilities
- **Key Functions**:
  - `compute_init()`: Initialize DDP, device, seeds
  - `compute_cleanup()`: Cleanup DDP process group
  - `get_dist_info()`: Get DDP rank/world_size
  - `autodetect_device_type()`: Auto-detect CUDA/MPS/CPU
- **Features**:
  - Automatic device detection
  - Reproducibility (seeds)
  - Precision settings (tf32 for CUDA)

### Reporting System (`nanochat/report.py`)
- **Purpose**: Generate training report cards
- **Key Features**:
  - System info collection (GPU, CPU, memory)
  - Git information tracking
  - Cost estimation
  - Bloat metrics (code size, dependencies)
  - Section-based logging
  - Final summary table generation
- **Key Classes**:
  - `Report`: Main report manager
  - `get_report()`: Convenience function (rank 0 only)
- **Report Sections**:
  - Tokenizer training/evaluation
  - Base model training/loss/evaluation
  - Midtraining
  - Chat evaluation (mid/sft/rl)
  - Chat SFT/RL

---

## 📈 Evaluation

### Loss Evaluation (`nanochat/loss_eval.py`)
- **Purpose**: Evaluate bits per byte (bpb) metric
- **Key Function**: `evaluate_bpb()`
- **Features**:
  - Tokenization-independent metric
  - Normalizes by byte length of tokens
  - Excludes special tokens
  - Handles ignore_index (-1) for masked tokens

### CORE Evaluation (`nanochat/core_eval.py`)
- **Purpose**: Evaluate CORE metric (DCLM paper)
- **Key Features**:
  - Multiple task types: multiple_choice, schema, language_modeling
  - Few-shot prompting support
  - Distributed evaluation
- **Key Functions**:
  - `evaluate_task()`: Evaluate a single task
  - `evaluate_example()`: Evaluate a single example
  - `forward_model()`: Get losses and predictions
- **Task Rendering**:
  - `render_prompts_mc()`: Multiple choice prompts
  - `render_prompts_schema()`: Schema prompts
  - `render_prompts_lm()`: Language modeling prompts

### Chat Evaluation (`scripts/chat_eval.py`)
- **Purpose**: Evaluate chat models on various tasks
- **Key Features**:
  - Generative evaluation (sampling)
  - Categorical evaluation (logits)
  - Distributed evaluation
  - ChatCORE metric calculation
- **Key Functions**:
  - `run_generative_eval()`: Sample-based evaluation
  - `run_categorical_eval()`: Logit-based evaluation
  - `run_chat_eval()`: Main evaluation dispatcher
- **Supported Tasks**: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SpellingBee

---

## 🔧 Supporting Components

### Common Utilities (`nanochat/common.py`)
- **Key Functions**:
  - `get_base_dir()`: Get nanochat cache directory
  - `print0()`: Print only on rank 0
  - `print_banner()`: ASCII art banner
  - `download_file_with_lock()`: Thread-safe file download
  - `DummyWandb`: No-op wandb for non-logging runs

### Configurator (`nanochat/configurator.py`)
- **Purpose**: Superior alternative to argparse
- **Features**: CLI argument parsing, config file support

### Execution (`nanochat/execution.py`)
- **Purpose**: Python code execution for tool use
- **Status**: Referenced but implementation not shown

---

## 🛠️ Rust Components

### rustbpe (`rustbpe/`)
- **Purpose**: Fast BPE tokenizer training in Rust
- **Key Features**:
  - GPT-4 style tokenizer training
  - Exports to tiktoken format
  - Much faster than Python implementations
- **Build**: `maturin develop --release`

---

## 📝 Scripts Overview

### Training Scripts
- `scripts/tok_train.py`: Train BPE tokenizer
- `scripts/tok_eval.py`: Evaluate tokenizer compression
- `scripts/base_train.py`: Base model pretraining
- `scripts/base_loss.py`: Evaluate base model loss
- `scripts/base_eval.py`: Evaluate base model (CORE metric)
- `scripts/mid_train.py`: Midtraining
- `scripts/chat_sft.py`: Supervised fine-tuning
- `scripts/chat_rl.py`: Reinforcement learning

### Evaluation Scripts
- `scripts/chat_eval.py`: Evaluate chat models
- `scripts/chat_cli.py`: CLI chat interface
- `scripts/chat_web.py`: Web UI chat interface

---

## 🔄 Training Pipeline Flow (from speedrun.sh)

```
1. Setup
   ├── Environment (uv, venv, wandb)
   ├── Rust/Cargo installation
   └── rustbpe build

2. Tokenizer Phase
   ├── Download dataset shards (8 initially, 240 total)
   ├── Train tokenizer (vocab_size=65536)
   └── Evaluate tokenizer

3. Base Model Phase
   ├── Download remaining dataset shards (240 total)
   ├── Pretrain model (d20 = 561M params)
   ├── Evaluate loss (bits per byte)
   └── Evaluate CORE metric

4. Midtraining Phase
   ├── Download identity conversations
   ├── Train on conversation format + tools + MC
   └── Evaluate on chat tasks

5. SFT Phase
   ├── Train on domain-specific tasks
   └── Evaluate on MMLU, ARC-Easy

6. Optional RL Phase
   ├── Reinforcement learning
   └── Evaluate on GSM8K

7. Reporting
   └── Generate comprehensive report.md
```

---

## 📚 Key File Locations

### Core Model
- `nanochat/gpt.py`: GPT model architecture
- `nanochat/engine.py`: Inference engine with KV cache
- `nanochat/tokenizer.py`: Tokenizer implementation

### Training
- `scripts/base_train.py`: Base pretraining
- `scripts/mid_train.py`: Midtraining
- `scripts/chat_sft.py`: Supervised fine-tuning
- `scripts/chat_rl.py`: Reinforcement learning

### Optimizers
- `nanochat/muon.py`: Muon optimizer
- `nanochat/adamw.py`: Distributed AdamW

### Data
- `nanochat/dataset.py`: Dataset download/management
- `nanochat/dataloader.py`: Data loading and tokenization
- `tasks/`: Evaluation task implementations

### Infrastructure
- `nanochat/checkpoint_manager.py`: Checkpoint save/load
- `nanochat/common.py`: Common utilities and DDP setup
- `nanochat/report.py`: Report generation

### Evaluation
- `nanochat/loss_eval.py`: Bits per byte evaluation
- `nanochat/core_eval.py`: CORE metric evaluation
- `scripts/chat_eval.py`: Chat model evaluation

---

## 🎓 Learning Path

1. **Start Here**: `speedrun.sh` - See the full pipeline
2. **Model Architecture**: `nanochat/gpt.py` - Understand the transformer
3. **Training Loop**: `scripts/base_train.py` - See how training works
4. **Data Flow**: `nanochat/dataloader.py` - Understand data loading
5. **Inference**: `nanochat/engine.py` - See how generation works
6. **Evaluation**: `scripts/chat_eval.py` - Understand evaluation metrics

---

*Generated for nanochat codebase analysis*

