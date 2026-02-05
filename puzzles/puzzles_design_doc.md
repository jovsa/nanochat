# Learning Curriculum for nanochat: Deep Coding Puzzles

## Goal
8 massive, expert-level puzzles to master nanochat internals.
**Constraint**: No 3-star puzzles. All puzzles involve building functional systems, not just calculations.

---

## Design Principles

1.  **Composite Challenges**: Each puzzle merges multiple concepts (e.g., "The Data Engine" = Loading + Packing + Mixing).
2.  **Build Functional Systems**: Output must be a working class or module (e.g., a DataLoader, an Optimizer, an Engine).
3.  **Verify Against Reality**: Use `nanochat` code as the ground truth oracle.
4.  **Standard Header**: Every puzzle file must start with a docstring explaining how to run, verify, and peek.
    ```python
    """
    Puzzle X: [Name]
    Goal: [Short Goal]

    Usage (from repo root):
    python -m puzzles.0X_name          # Run the puzzle
    python -m puzzles.0X_name --verify # Verify your solution
    python -m puzzles.0X_name --peek   # See reference solution (oracle)
    """
    ```

---

## The 8 Puzzles

### Puzzle 1: The Transformer Blueprint ⭐⭐⭐⭐

**Goal**: Build the complete static graph setup for GPT.
**Merged from**: Architecture Scaling, Sliding Window, RoPE setup.

**Your Task**: Implement a `ModelSetup` class that:
1.  Derives `GPTConfig` from `depth` (scaling laws logic).
2.  Computes RoPE frequencies (`cos`, `sin`) with proper dtype handling.
3.  Generates the Sliding Window Attention mask for *any* pattern (e.g., "SSSL").

**Verification**: Instantiates `nanochat.gpt.GPT` and compares `model.params`, `model.cos`, `model.sin`, and attention masks.

---

### Puzzle 2: The Optimization Layer ⭐⭐⭐⭐⭐

**Goal**: Build the training stepper.
**Merged from**: Muon Optimizer, FP8 Training, LR Schedules.

**Your Task**: Implement an `OptimizerFactory` that:
1.  Splits params into 3 groups: `muon_matrix` (2D), `adam_decay`, `adam_no_decay`.
2.  Implements the Newton-Schulz update step (Muon).
3.  Implements the `fp8_module_filter` and context manager for safe FP8 fallback.
4.  Derives the warm-up/warm-down LR schedule.

**Verification**: Runs a dummy training step and compares weight updates against `nanochat.optim.Muon`.

---

### Puzzle 3: The Data Engine ⭐⭐⭐⭐

**Goal**: Build a high-performance data pipeline for SFT and Pretraining.
**Merged from**: BOS Dataloader, SFT Data Mixing.

**Your Task**: Implement a `UniversalDataLoader` that handles:
1.  **Sharding**: Parquet reading with DDP awareness.
2.  **Packing**: Best-fit bin packing algorithm (SFT style) AND BOS-alignment (Pretrain style).
3.  **Mixing**: `TaskMixture` logic for combining datasets (e.g. SmolTalk + MMLU).

**Verification**: Generates a batch of correctly packed, masked, and mixed tensors matching `scripts/chat_sft.py`.

---

### Puzzle 4: The Inference Engine ⭐⭐⭐⭐⭐

**Goal**: Build the streaming generation loop with tool use.
**Merged from**: KV Cache, Inference Engine.

**Your Task**: Implement `Engine.generate()` with:
1.  **KV Cache**: Manage `(B,T,H,D)` layout and `cache_seqlens`.
2.  **Tool State Machine**: Detect `<|python_start|>`, pause, execute (mocked), resume.
3.  **Forced Tokens**: Inject tool outputs into the stream.
4.  **Sampling**: Temp/Top-k logic.

**Verification**: Runs a generation loop that correctly "uses" a calculator tool and produces expected tokens.

---

### Puzzle 5: The RL Loop ⭐⭐⭐⭐⭐

**Goal**: Build the complete GRPO training loop.
**Merged from**: Simplified GRPO, Full GRPO.

**Your Task**: Implement the `RLTrainer` class:
1.  **Rollout**: Generate `K` samples with `temperature=1.0`.
2.  **Advantage**: Compute `(r - mean)` (Simplified) OR `(r - mean)/std` (Full).
3.  **Loss**: Compute token-level policy gradient with masking.
4.  **KL Penalty**: (Optional) Add KL div against a reference model.

**Verification**: Computes gradients on a dummy batch and matches `scripts/chat_rl.py`.

---

### Puzzle 6: The Evaluator ⭐⭐⭐⭐

**Goal**: Build the evaluation system.
**Merged from**: Custom Eval Task, CORE Metric.

**Your Task**: Implement `CORE_Evaluator`:
1.  **Batching**: Group MC options by common prefix/suffix.
2.  **Rendering**: Jinja2 logic for MC/Schema/LM tasks.
3.  **Scoring**: Loss-based selection logic.
4.  **Aggregation**: DDP reduction of results.

**Verification**: Evaluates a mock "MMLU" question and matches `nanochat.core_eval` results.

---

### Puzzle 7: The Sandbox & Tokenizer ⭐⭐⭐⭐⭐

**Goal**: Build the safety and text processing layer.
**Merged from**: Custom Tokenizer, Sandboxed Execution.

**Your Task**:
1.  **Tokenizer**: Implement `RustBPETokenizer` wrapper with logic for `<|python_start|>` injection.
2.  **Sandbox**: Implement `execute_code` with `multiprocessing` isolation and resource limits.
3.  **Security Test**: Pass a suite of "bad actor" scripts (fork bombs, file deletion).

**Verification**: Successfully tokenizes tool-use conversations and safely executes code.

---

### Puzzle 8: Mini GPT from Scratch ⭐⭐⭐⭐⭐

**Goal**: Build the model itself.
**Merged from**: Mini GPT.

**Your Task**: Implement `MiniGPT` (Transformer):
1.  RMSNorm, RoPE, QK Norm.
2.  GQA Attention with Flash Attention calls.
3.  MLP with `ReLU^2` activation.
4.  ResFormer value residual connection.

**Verification**: Loads weights from a real nanochat checkpoint and achieves identical forward pass outputs.

---

## Verification Strategy

**Philosophy: "The Code is the Solution"**

Since these puzzles are about learning *this specific repo*, the solution **IS** the existing `nanochat` code.

For every puzzle:
1.  **The Oracle**: The test suite imports the actual `nanochat` modules as the ground truth.
2.  **The Challenge**: You implement a feature (e.g., `ModelSetup`) to match the behavior of the existing codebase.
3.  **The Solution**: If you are stuck, the `peek()` function will output the **actual source code lines** from the repository that implement the logic.
    - Example: `python -m puzzles.01_transformer_blueprint --peek` (from repo root) will print lines from `nanochat/gpt.py` and `scripts/base_train.py`.

**Verification**: 
We use the `nanochat` codebase as the ground truth oracle.
`torch.allclose(your_output, nanochat_output)` means you have successfully reverse-engineered the logic.
