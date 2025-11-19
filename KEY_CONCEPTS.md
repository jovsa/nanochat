# Key Concepts to Master nanochat

This document outlines the essential concepts you need to understand to master every aspect of the nanochat codebase.

---

## 1. Transformer Architecture

### Rotary Position Embeddings (RoPE)
- **What**: Relative positional encoding using rotations in the complex plane
- **Why**: More efficient than absolute positional embeddings, better generalization
- **Where**: `nanochat/gpt.py` - `_precompute_rotary_embeddings()`, `apply_rotary_emb()`
- **Key Details**:
  - Applied to queries and keys before attention
  - Precomputed for efficiency
  - Base frequency: 10000 (configurable)
  - Stored in bfloat16 to save memory

### Group Query Attention (GQA)
- **What**: Attention mechanism where multiple query heads share key/value heads
- **Why**: Reduces memory and computation for inference (especially with KV cache)
- **Where**: `nanochat/gpt.py` - `CausalSelfAttention` class
- **Key Details**:
  - `n_head`: Number of query heads
  - `n_kv_head`: Number of key/value heads (usually ≤ n_head)
  - When `n_head == n_kv_head`: Standard multi-head attention
  - When `n_head > n_kv_head`: GQA (keys/values are duplicated/broadcast)

### QK Norm
- **What**: Normalization applied to queries and keys before attention
- **Why**: Stabilizes training, improves performance
- **Where**: `nanochat/gpt.py` - `CausalSelfAttention.forward()`
- **Key Details**:
  - Uses RMSNorm (no learnable parameters)
  - Applied after rotary embeddings
  - Formula: `norm(q), norm(k)` where `norm` is RMSNorm

### ReLU² Activation
- **What**: Squared ReLU activation: `relu(x)²`
- **Why**: Better performance than standard ReLU for some architectures
- **Where**: `nanochat/gpt.py` - `MLP.forward()`
- **Key Details**:
  - Applied in the MLP feed-forward network
  - Formula: `F.relu(x).square()`

### RMSNorm (Root Mean Square Normalization)
- **What**: Normalization without learnable parameters
- **Why**: Simpler, more efficient than LayerNorm
- **Where**: `nanochat/gpt.py` - `norm()` function
- **Key Details**:
  - Uses PyTorch's `F.rms_norm()`
  - Applied after attention and MLP (pre-norm architecture)
  - No bias or scale parameters

### KV Cache for Efficient Inference
- **What**: Caching key/value states to avoid recomputation during autoregressive generation
- **Why**: Dramatically speeds up generation (only compute new tokens, not entire sequence)
- **Where**: `nanochat/engine.py` - `KVCache` class
- **Key Details**:
  - Shape: `(num_layers, 2, batch_size, num_heads, seq_len, head_dim)`
  - Dynamically grows as sequence length increases
  - Supports prefill (batch 1) + decode (batch N) pattern
  - Position tracking for rotary embedding offset

---

## 2. Training Techniques

### Distributed Data Parallel (DDP)
- **What**: PyTorch's distributed training across multiple GPUs
- **Why**: Parallelize training across devices, reduce training time
- **Where**: `nanochat/common.py` - `compute_init()`, all training scripts
- **Key Details**:
  - Each GPU processes different data shards
  - Gradients are averaged across ranks
  - Uses NCCL backend for CUDA
  - Rank 0 handles logging, checkpointing

### Gradient Accumulation
- **What**: Accumulate gradients over multiple micro-batches before optimizer step
- **Why**: Simulate larger batch sizes when memory is limited
- **Where**: All training scripts (e.g., `scripts/base_train.py`)
- **Key Details**:
  - Loss is divided by `grad_accum_steps` before backward
  - Gradients are summed automatically
  - Formula: `total_batch_size = device_batch_size * world_size * grad_accum_steps`

### Mixed Precision Training
- **What**: Use bfloat16 for forward/backward, float32 for optimizer states
- **Why**: Faster training, lower memory usage
- **Where**: All training scripts - `torch.amp.autocast()`
- **Key Details**:
  - bfloat16 for activations and weights
  - float32 for loss computation and optimizer states
  - Automatic loss scaling (handled by PyTorch)

### Learning Rate Scheduling
- **What**: Adjust learning rate during training
- **Why**: Better convergence, avoid overshooting
- **Where**: Training scripts - `get_lr_multiplier()` functions
- **Key Patterns**:
  - **Warmup**: Linear increase from 0 to target LR
  - **Constant**: Maintain target LR
  - **Warmdown**: Linear decrease to final LR
  - **Cosine/Linear decay**: Various decay schedules

### Gradient Clipping
- **What**: Clip gradients to prevent explosion
- **Why**: Stabilize training, prevent NaN
- **Where**: Training scripts - `torch.nn.utils.clip_grad_norm_()`
- **Key Details**:
  - Clips gradient norm to specified value (e.g., 1.0)
  - Applied before optimizer step
  - Helps with training stability

---

## 3. Optimizers

### Muon Optimizer
- **What**: SGD-momentum with Newton-Schulz orthogonalization
- **Why**: Better optimization for 2D matrix parameters (linear layers)
- **Where**: `nanochat/muon.py`
- **Key Details**:
  - **Internal**: Runs SGD-momentum (with optional Nesterov)
  - **Post-processing**: Orthogonalizes the update using Newton-Schulz iteration
  - **Aspect-ratio scaling**: Learning rate scaled by `sqrt(max(H, W) / min(H, W))`
  - **Usage**: Applied to transformer blocks (attention + MLP linear layers)
  - **Not for**: Embeddings, lm_head, or 0D/1D parameters

### Newton-Schulz Iteration
- **What**: Iterative method to compute matrix square root / orthogonalization
- **Why**: Efficiently orthogonalize updates in bfloat16
- **Where**: `nanochat/muon.py` - `zeropower_via_newtonschulz5()`
- **Key Details**:
  - Quintic iteration (5 steps typically)
  - Coefficients: `(3.4445, -4.7750, 2.0315)`
  - Produces approximately orthogonal updates
  - Stable in bfloat16 precision

### Distributed AdamW
- **What**: AdamW optimizer with ZeRO-2 style sharding
- **Why**: Reduce memory usage, enable larger models
- **Where**: `nanochat/adamw.py` - `DistAdamW`
- **Key Details**:
  - **Gradients**: Reduce-scatter (average and shard)
  - **Parameters**: Sharded across ranks
  - **States**: Sharded (exp_avg, exp_avg_sq per rank)
  - **Synchronization**: All-gather after update
  - **Usage**: Applied to embeddings and lm_head

### Parameter Group Separation
- **What**: Different optimizers/learning rates for different parameter types
- **Why**: Different parameters benefit from different optimization strategies
- **Where**: `nanochat/gpt.py` - `setup_optimizers()`
- **Key Details**:
  - **Matrix params** (transformer blocks): Muon optimizer
  - **Embedding params** (wte): AdamW optimizer
  - **Unembedding params** (lm_head): AdamW optimizer
  - **Learning rates**: Different LR for each group (scaled by model dimension)

---

## 4. Tokenization

### BPE (Byte Pair Encoding)
- **What**: Subword tokenization algorithm
- **Why**: Balance between word-level and character-level tokenization
- **Where**: `nanochat/tokenizer.py`, `rustbpe/`
- **Key Details**:
  - Starts with byte-level tokens (256 tokens)
  - Iteratively merges most frequent pairs
  - Vocab size: 65536 (2^16) in nanochat
  - GPT-4 style pre-tokenization regex pattern

### GPT-4 Style Tokenizer
- **What**: Specific pre-tokenization pattern used by GPT-4
- **Why**: Better tokenization quality, matches GPT-4 behavior
- **Where**: `nanochat/tokenizer.py` - `SPLIT_PATTERN`
- **Key Details**:
  - Regex pattern splits text before BPE
  - Handles contractions, numbers, punctuation
  - Modified from GPT-4: `\p{N}{1,2}` instead of `\p{N}{1,3}` (for smaller vocab)

### Special Tokens
- **What**: Reserved tokens for conversation structure and tool use
- **Why**: Enable structured conversations and tool calling
- **Where**: `nanochat/tokenizer.py` - `SPECIAL_TOKENS`
- **Key Tokens**:
  - `<|bos|>`: Beginning of sequence (document delimiter)
  - `<|user_start|>`, `<|user_end|>`: User message boundaries
  - `<|assistant_start|>`, `<|assistant_end|>`: Assistant message boundaries
  - `<|python_start|>`, `<|python_end|>`: Python tool call boundaries
  - `<|output_start|>`, `<|output_end|>`: Tool output boundaries

### Conversation Rendering
- **What**: Convert conversation dict to token IDs with supervision mask
- **Why**: Train only on assistant tokens, not user/tool tokens
- **Where**: `nanochat/tokenizer.py` - `render_conversation()`
- **Key Details**:
  - Returns `(ids, mask)` where mask=1 for supervised tokens
  - User messages: mask=0 (not supervised)
  - Assistant messages: mask=1 (supervised)
  - Tool calls: mask=1 (supervised)
  - Tool outputs: mask=0 (not supervised, comes from execution)

---

## 5. Data Pipeline

### Parquet File Format
- **What**: Columnar storage format (Apache Parquet)
- **Why**: Efficient storage and reading of large datasets
- **Where**: `nanochat/dataset.py`
- **Key Details**:
  - Each file contains multiple row groups
  - Text stored in 'text' column
  - Row groups are sharded across DDP ranks
  - Files are downloaded on-demand

### Streaming Data Loading
- **What**: Load and tokenize data on-the-fly
- **Why**: Avoid loading entire dataset into memory
- **Where**: `nanochat/dataloader.py`
- **Key Details**:
  - Infinite iterator (multi-epoch)
  - Tokenization happens in batches
  - Token buffer accumulates until batch size is reached
  - Memory pinning for faster CPU→GPU transfer

### Distributed Data Sharding
- **What**: Each DDP rank processes different data shards
- **Why**: Avoid data duplication, parallelize data loading
- **Where**: `nanochat/dataloader.py` - `tokenizing_distributed_data_loader_with_state()`
- **Key Details**:
  - Row groups are assigned by rank: `rg_idx = base_idx * world_size + rank`
  - Each rank processes every `world_size`-th row group
  - State dict tracks position for approximate resume

### Task Mixture
- **What**: Combine multiple datasets into one training set
- **Why**: Train on diverse tasks simultaneously
- **Where**: `tasks/common.py` - `TaskMixture`
- **Key Details**:
  - Deterministic shuffle ensures task mixing
  - Tasks can be oversampled (add multiple times)
  - Supports slicing (start, stop, step)

---

## 6. Evaluation Metrics

### Bits Per Byte (bpb)
- **What**: Tokenization-independent loss metric
- **Why**: Compare models with different vocab sizes fairly
- **Where**: `nanochat/loss_eval.py` - `evaluate_bpb()`
- **Key Details**:
  - Formula: `total_nats / (log(2) * total_bytes)`
  - Normalizes loss by byte length of tokens
  - Excludes special tokens (byte length = 0)
  - Handles ignore_index (-1) for masked tokens

### CORE Metric
- **What**: Comprehensive evaluation across multiple tasks (DCLM paper)
- **Why**: Single metric to compare base models
- **Where**: `nanochat/core_eval.py`, `scripts/base_eval.py`
- **Key Details**:
  - Mean of centered accuracies across tasks
  - Tasks: multiple choice, schema, language modeling
  - Few-shot prompting support
  - Formula: `mean((acc - baseline) / (1 - baseline))`

### ChatCORE Metric
- **What**: CORE metric for chat models
- **Why**: Single metric to compare chat models
- **Where**: `scripts/chat_eval.py`
- **Key Details**:
  - Similar to CORE but for chat tasks
  - Tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval
  - Baseline accuracies: 25% for MC, 0% for generative
  - Formula: `mean((acc - baseline) / (1 - baseline))`

### Task-Specific Metrics
- **MMLU**: Multiple choice accuracy across 57 subjects
- **ARC**: Science question accuracy (Easy/Challenge)
- **GSM8K**: Math problem solving accuracy
- **HumanEval**: Python code generation pass rate
- **SpellingBee**: Spelling/counting accuracy

---

## 7. Advanced Concepts

### Approximate Resume
- **What**: Resume training from approximate checkpoint
- **Why**: Continue training after interruption without perfect state
- **Where**: `nanochat/dataloader.py` - `resume_state_dict`
- **Key Details**:
  - Tracks parquet file index and row group index
  - May skip a few documents (acceptable trade-off)
  - Perfect resume would require more complex state tracking

### Model Compilation
- **What**: Compile model with `torch.compile()` for faster execution
- **Why**: Significant speedup (especially on newer PyTorch versions)
- **Where**: Training scripts - `torch.compile(model, dynamic=False)`
- **Key Details**:
  - `dynamic=False`: Input shapes don't change (safe for training)
  - `dynamic=True`: Input shapes vary (needed for variable-length sequences)
  - Original model kept for checkpointing (compiled model may change shapes)

### Meta Device Initialization
- **What**: Initialize model on "meta" device (no memory allocation)
- **Why**: Initialize large models without allocating memory
- **Where**: `scripts/base_train.py` - `torch.device("meta")`
- **Key Details**:
  - Creates model structure without weights
  - Weights initialized after moving to actual device
  - Useful for very large models

### Logits Softcap
- **What**: Apply tanh-based softcap to logits
- **Why**: Prevent extreme logit values, stabilize training
- **Where**: `nanochat/gpt.py` - `forward()` method
- **Key Details**:
  - Formula: `softcap * tanh(logits / softcap)`
  - Softcap value: 15 (hardcoded)
  - Applied before cross-entropy loss

### Tool Use (Python Calculator)
- **What**: Model can execute Python expressions as a tool
- **Why**: Enable mathematical reasoning and string operations
- **Where**: `nanochat/engine.py` - `use_calculator()`
- **Key Details**:
  - Safe evaluation with timeout (3 seconds)
  - Supports math expressions and string operations (`.count()`)
  - Disallows dangerous patterns (import, exec, etc.)
  - Special tokens: `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`

---

## 8. Training Phases

### Phase 1: Base Pretraining
- **Data**: Raw text from FineWeb-Edu
- **Objective**: Next token prediction (language modeling)
- **Metrics**: Bits per byte, CORE metric
- **Duration**: ~4 hours for d20 model on 8XH100

### Phase 2: Midtraining
- **Data**: Task mixture (conversations, MC, math, spelling)
- **Objective**: Learn conversation format, tool use, multiple choice
- **Metrics**: Validation bpb, task accuracies
- **Duration**: ~10 iterations (configurable)

### Phase 3: Supervised Fine-tuning
- **Data**: Curated task mixture (smaller, higher quality)
- **Objective**: Domain adaptation to specific tasks
- **Metrics**: Task accuracies (MMLU, ARC, GSM8K)
- **Duration**: 1 epoch (configurable)

### Phase 4: Reinforcement Learning (Optional)
- **Data**: GSM8K problems
- **Objective**: Improve math reasoning via RL
- **Metrics**: GSM8K accuracy
- **Status**: Experimental

---

## 9. Memory and Performance

### Gradient Checkpointing
- **Status**: Not currently used in nanochat
- **Alternative**: Smaller batch sizes, gradient accumulation

### Mixed Precision
- **Activation Precision**: bfloat16
- **Weight Precision**: bfloat16 (embeddings), bfloat16 (transformer)
- **Optimizer Precision**: float32 (AdamW states), bfloat16 (Muon)

### Batch Size Management
- **Device Batch Size**: Per-GPU batch size (limited by VRAM)
- **Total Batch Size**: Global batch size in tokens
- **Gradient Accumulation**: Automatically calculated to reach total batch size

### Memory Optimization
- **torch.compile()**: Reduces memory usage via fusion
- **Gradient accumulation**: Reduces peak memory
- **Mixed precision**: Reduces activation memory
- **KV cache**: Efficient inference memory usage

---

## 10. Code Organization Principles

### Minimalism
- **Philosophy**: Keep codebase small and readable
- **No**: Giant config objects, model factories, if-then-else monsters
- **Yes**: Single cohesive baseline, hackable, forkable

### End-to-End Pipeline
- **Philosophy**: Single script runs everything
- **Example**: `speedrun.sh` trains from scratch to chat model
- **Benefits**: Reproducibility, ease of use

### Distributed-First Design
- **Philosophy**: All code works in both single-GPU and multi-GPU settings
- **Implementation**: DDP-aware data loading, gradient synchronization
- **Fallback**: Automatically uses gradient accumulation if single GPU

---

## Learning Resources

1. **Transformer Architecture**: "Attention Is All You Need" (Vaswani et al., 2017)
2. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
3. **GQA**: "GQA: Training Generalized Query Transformer Models" (Ainslie et al., 2023)
4. **Muon Optimizer**: https://kellerjordan.github.io/posts/muon/
5. **CORE Metric**: "Direct Language Model Alignment" (DCLM paper, 2024)
6. **Chinchilla Scaling**: "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)

---

*This document covers the essential concepts needed to master the nanochat codebase. Study these concepts alongside the code to build deep understanding.*

