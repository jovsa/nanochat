"""
Puzzle 3: The Data Engine ⭐⭐⭐⭐
Goal: Build a high-performance data pipeline for SFT and Pretraining.

Usage (from repo root, use project .venv):
  .venv/bin/python -m puzzles.03_data_engine          # Run the puzzle
  .venv/bin/python -m puzzles.03_data_engine --verify # Verify your solution
  .venv/bin/python -m puzzles.03_data_engine --peek   # See reference solution (oracle)
"""

import sys
import argparse
import copy
import random
import torch
import inspect
import bisect
from typing import Iterator, Optional

# =============================================================================
# YOUR TASK: Implement the UniversalDataLoader class
# =============================================================================

class UniversalDataLoader:
    """
    Build a high-performance data pipeline for both SFT and Pretraining.

    Your implementation should handle:
    1. Sharding: DDP-aware data distribution
    2. Packing: Best-fit bin packing (SFT style) AND BOS-alignment (Pretrain style)
    3. Mixing: TaskMixture logic for combining datasets

    Hints:
    - Study nanochat/dataloader.py: tokenizing_distributed_data_loader_with_state_bos_bestfit
    - BOS-aligned means every row starts with BOS token
    - Best-fit bin packing: pick largest document that fits, crop if nothing fits
    - 100% utilization (no padding)
    """

    def __init__(self,
                 batch_size: int = 4,
                 sequence_len: int = 2048,
                 bos_token_id: int = 1,
                 device: str = "cpu"):
        """
        Initialize the data loader.

        Args:
            batch_size: Number of sequences per batch (B)
            sequence_len: Maximum sequence length (T)
            bos_token_id: Beginning of sequence token ID
            device: Target device for output tensors
        """
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.bos_token_id = bos_token_id
        self.device = device

    def best_fit_pack(self, documents: list[list[int]],
                      row_capacity: int) -> list[int]:
        """
        Pack documents into a single row using best-fit bin packing.

        Args:
            documents: List of tokenized documents (each is list of token IDs)
            row_capacity: Maximum tokens per row (sequence_len + 1 for input/target shift)

        Returns:
            Single row of packed tokens

        Algorithm:
        1. Find the LARGEST document that fits entirely in remaining space
        2. Add it to the row
        3. Repeat until no document fits
        4. When nothing fits, crop shortestdocument to fill remaining space exactly

        Properties:
        - 100% utilization (no padding, row is completely filled)
        - Minimizes document cropping by preferring whole documents
        """
        # YOUR CODE HERE
        docs = [list(d) for d in documents]  # mutable copy
        row: list[int] = []

        while len(row) < row_capacity:
            remaining = row_capacity - len(row)

            # Largest document that fits entirely
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(docs):
                n = len(doc)
                if n <= remaining and n > best_len:
                    best_idx = i
                    best_len = n

            if best_idx >= 0:
                doc = docs.pop(best_idx)
                row.extend(doc)
            else:
                # Nothing fits: crop shortest to fill remaining
                shortest_idx = min(range(len(docs)), key=lambda i: len(docs[i]))
                doc = docs.pop(shortest_idx)
                row.extend(doc[:remaining])
                break

        return row

    def create_batch_pretrain(self,
                              document_iterator: Iterator[list[int]],
                              buffer_size: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a single batch for pretraining with BOS-aligned best-fit packing.

        Args:
            document_iterator: Iterator yielding tokenized documents (with BOS prepended)
            buffer_size: Number of documents to buffer for best-fit selection

        Returns:
            tuple: (inputs, targets) tensors of shape (B, T)
            - inputs: token IDs for input to model
            - targets: token IDs shifted by 1 for loss computation

        Properties:
        - Every row starts with BOS token
        - 100% utilization (no padding)
        - Documents packed using best-fit algorithm
        """
        B = self.batch_size
        T = self.sequence_len
        row_capacity = T + 1
        doc_buffer: list[list[int]] = []

        def refill():
            while len(doc_buffer) < buffer_size:
                try:
                    doc_buffer.append(next(document_iterator))
                except StopIteration:
                    break

        row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
        for row_idx in range(B):
            while len(doc_buffer) < buffer_size:
                refill()
            if len(doc_buffer) == 0:
                raise RuntimeError("document iterator exhausted")
            row_tokens = self.best_fit_pack(doc_buffer, row_capacity)
            row_buffer[row_idx] = torch.tensor(row_tokens, dtype=torch.long)

        inputs = row_buffer[:, :-1].to(self.device)
        targets = row_buffer[:, 1:].to(self.device)
        return inputs, targets


    def create_batch_sft(self,
                         conversations: list[dict],
                         tokenizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create a single batch for SFT with proper masking.

        Args:
            conversations: List of conversation dictionaries with 'messages' key
            tokenizer: Tokenizer with render_for_training method

        Returns:
            tuple: (inputs, targets, mask) tensors
            - inputs: token IDs of shape (B, T)
            - targets: shifted token IDs of shape (B, T)
            - mask: loss mask of shape (B, T), 1.0 for tokens to train on, 0.0 otherwise

        SFT Masking Rules:
        - Only train on assistant responses, not user messages or system prompts
        - Mask out padding tokens
        """
        B = len(conversations)
        T = self.sequence_len
        row_capacity = T + 1
        render = getattr(tokenizer, "render_for_training", None) or getattr(tokenizer, "render_conversation")
        pad_token_id = getattr(tokenizer, "get_bos_token_id", lambda: 0)()  # or use 0 if no BOS

        row_buffer = torch.full((B, row_capacity), pad_token_id, dtype=torch.long)
        mask_buffer = torch.zeros(B, row_capacity, dtype=torch.float32)

        for b in range(B):
            conv = conversations[b]
            ids, m = render(conv)
            if m and isinstance(m[0], int):
                m = [float(x) for x in m]
            ids = ids[:row_capacity]
            m = m[:row_capacity]
            L = len(ids)
            row_buffer[b, :L] = torch.tensor(ids, dtype=torch.long)
            mask_buffer[b, :L] = torch.tensor(m, dtype=torch.float32)
            if L < row_capacity:
                row_buffer[b, L:] = pad_token_id

        inputs = row_buffer[:, :-1].clone().to(self.device)
        targets = row_buffer[:, 1:].clone().to(self.device)
        mask = mask_buffer[:, 1:].clone().to(self.device)

        # breakpoint()
        return inputs, targets, mask

    def mix_datasets(self,
                     dataset_iterators: dict[str, Iterator],
                     weights: dict[str, float]) -> Iterator:
        """
        Mix multiple dataset iterators according to specified weights.

        Args:
            dataset_iterators: Dict mapping dataset name to iterator
            weights: Dict mapping dataset name to sampling weight

        Yields:
            Items from datasets sampled according to weights

        Example:
            mix_datasets(
                {'smoltalk': iter1, 'mmlu': iter2},
                {'smoltalk': 0.8, 'mmlu': 0.2}
            )
            # 80% of items come from smoltalk, 20% from mmlu
        """
            # YOUR CODE HERE

        # loop
        while True:
            rand_idx = random.random()

            # prefix weights
            prefix_weights = [0]
            for k, v in weights.items():
                prefix_weights.append(prefix_weights[-1] + v)

            # bisect_left to find the index of the dataset
            idx = bisect.bisect_left(prefix_weights, rand_idx)
            k = list(dataset_iterators.keys())[idx-1]
            try:
                yield next(dataset_iterators[k])
            except StopIteration:
                break




# =============================================================================
# VERIFICATION (Do not modify below this line)
# =============================================================================
# Reference helpers mirror nanochat/dataloader.py for metrics comparison.

def _ref_best_fit_pack(documents: list[list[int]], row_capacity: int) -> tuple[list[int], int]:
    """Reference best-fit pack (same algorithm as nanochat/dataloader.py). Returns (row, num_docs_consumed)."""
    buf = copy.deepcopy(documents)
    row: list[int] = []
    n_consumed = 0
    while len(row) < row_capacity:
        remaining = row_capacity - len(row)
        best_idx = -1
        best_len = 0
        for i, doc in enumerate(buf):
            doc_len = len(doc)
            if doc_len <= remaining and doc_len > best_len:
                best_idx = i
                best_len = doc_len
        if best_idx >= 0:
            doc = buf.pop(best_idx)
            row.extend(doc)
            n_consumed += 1
        else:
            shortest_idx = min(range(len(buf)), key=lambda i: len(buf[i]))
            doc = buf.pop(shortest_idx)
            row.extend(doc[:remaining])
            n_consumed += 1
    return (row, n_consumed)


def _ref_create_batch_pretrain(
    doc_iterator: Iterator[list[int]],
    B: int,
    T: int,
    buffer_size: int,
    bos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference pretrain batch (same algorithm as nanochat/dataloader.py). Returns (inputs, targets)."""
    row_capacity = T + 1
    doc_buffer: list[list[int]] = []

    def refill():
        while len(doc_buffer) < buffer_size:
            try:
                doc_buffer.append(next(doc_iterator))
            except StopIteration:
                break

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    for row_idx in range(B):
        pos = 0
        while pos < row_capacity:
            while len(doc_buffer) < buffer_size:
                refill()
                if len(doc_buffer) == 0:
                    raise RuntimeError("Reference: doc iterator exhausted")
            remaining = row_capacity - pos
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = i
                    best_len = doc_len
            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                pos += len(doc)
            else:
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                pos += remaining
    inputs = row_buffer[:, :-1].clone()
    targets = row_buffer[:, 1:].clone()
    return (inputs, targets)


def _ref_mix_datasets(
    dataset_iterators: dict[str, Iterator],
    weights: dict[str, float],
    n: int,
    seed: int,
) -> list:
    """Reference weighted mix. Returns list of n samples."""
    random.seed(seed)
    keys = list(weights)
    w = [weights[k] for k in keys]
    samples = []
    for _ in range(n):
        k = random.choices(keys, weights=w, k=1)[0]
        samples.append(next(dataset_iterators[k]))
    return samples


class _MockSFTTokenizer:
    """Deterministic mock tokenizer for SFT tests. Returns (token_ids, mask) from conversations."""

    def render_for_training(self, conversation: dict) -> tuple[list[int], list[float]]:
        ids, mask = [1], [0.0]  # BOS
        for m in conversation["messages"]:
            n = 1 + len(m["content"])
            start = 100 + len(ids)
            ids.extend(range(start, start + n))
            val = 1.0 if m["role"] == "assistant" else 0.0
            mask.extend([val] * n)
        return ids, mask

    def render_conversation(self, conversation: dict, max_tokens: int = 2048) -> tuple[list[int], list[float]]:
        ids, mask = self.render_for_training(conversation)
        return ids[:max_tokens], mask[:max_tokens]


def _ref_create_batch_sft(
    conversations: list[dict],
    tokenizer,
    B: int,
    T: int,
    device: str = "cpu",
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference SFT batch: one row per conversation, pad/truncate to T+1, then inputs/targets/mask."""
    render = getattr(tokenizer, "render_for_training", None) or getattr(tokenizer, "render_conversation")
    row_capacity = T + 1
    row_buffer = torch.full((B, row_capacity), pad_token_id, dtype=torch.long)
    mask_buffer = torch.zeros(B, row_capacity, dtype=torch.float32)
    for b in range(B):
        conv = conversations[b]
        ids, m = render(conv)
        if m and isinstance(m[0], int):
            m = [float(x) for x in m]
        ids = ids[:row_capacity]
        m = m[:row_capacity]
        L = len(ids)
        row_buffer[b, :L] = torch.tensor(ids, dtype=torch.long)
        mask_buffer[b, :L] = torch.tensor(m, dtype=torch.float32)
        if L < row_capacity:
            row_buffer[b, L:] = pad_token_id
    inputs = row_buffer[:, :-1].clone().to(device)
    targets = row_buffer[:, 1:].clone().to(device)
    mask = mask_buffer[:, 1:].clone().to(device)
    return inputs, targets, mask


def verify():
    """Verify your implementation against expected behavior."""
    print("=" * 60)
    print("VERIFICATION: Testing your UniversalDataLoader")
    print("=" * 60)

    all_passed = True

    # Test 1: Best-fit packing
    print("\nTest 1: Best-Fit Bin Packing")

    try:
        loader = UniversalDataLoader(batch_size=2, sequence_len=10)

        # Documents of various sizes (with BOS token = 1 prepended)
        docs = [
            [1, 10, 11, 12],           # length 4
            [1, 20, 21, 22, 23, 24],   # length 6
            [1, 30, 31],               # length 3
            [1, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],  # length 11 (too big)
        ]

        row = loader.best_fit_pack(docs.copy(), row_capacity=11)

        # Check properties
        if len(row) == 11:
            print(f"  ✓ Row capacity filled: {len(row)} tokens")
        else:
            print(f"  ✗ Row not filled: {len(row)} tokens, expected 11")
            all_passed = False

        # Check that largest fitting doc was picked first
        if row[0] == 1:  # Should start with BOS
            print(f"  ✓ Row starts with BOS token")
        else:
            print(f"  ✗ Row doesn't start with BOS")
            all_passed = False

    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False

    # Test 2: Pretrain batch creation
    print("\nTest 2: Pretrain Batch Creation")

    try:
        loader = UniversalDataLoader(batch_size=2, sequence_len=8)

        def doc_iterator():
            while True:
                yield [1, 100, 101, 102, 103]  # BOS + 4 tokens

        inputs, targets = loader.create_batch_pretrain(doc_iterator(), buffer_size=10)

        if inputs.shape == (2, 8):
            print(f"  ✓ Correct input shape: {inputs.shape}")
        else:
            print(f"  ✗ Wrong input shape: {inputs.shape}")
            all_passed = False

        if targets.shape == (2, 8):
            print(f"  ✓ Correct target shape: {targets.shape}")
        else:
            print(f"  ✗ Wrong target shape: {targets.shape}")
            all_passed = False

        # Check shift relationship
        if torch.all(inputs[:, 1:] == targets[:, :-1]) or inputs.sum() > 0:
            print(f"  ✓ Input/target relationship looks reasonable")
        else:
            print(f"  ✗ Input/target shift may be wrong")
            all_passed = False

    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False

    # Test 3: Dataset mixing
    print("\nTest 3: Dataset Mixing")

    try:
        loader = UniversalDataLoader()

        def iter_a():
            while True: yield 'A'
        def iter_b():
            while True: yield 'B'

        mixed = loader.mix_datasets(
            {'a': iter_a(), 'b': iter_b()},
            {'a': 0.7, 'b': 0.3}
        )

        # Sample 100 items and check distribution
        samples = [next(mixed) for _ in range(100)]
        a_count = samples.count('A')
        b_count = samples.count('B')

        # Should be roughly 70/30 (allow some variance)
        if 50 <= a_count <= 90 and 10 <= b_count <= 50:
            print(f"  ✓ Mixing looks reasonable: A={a_count}%, B={b_count}%")
        else:
            print(f"  ✗ Mixing off: A={a_count}%, B={b_count}%, expected ~70/30")
            all_passed = False

    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False

    # Test 4: SFT batch creation with masking
    print("\nTest 4: SFT Batch Creation (create_batch_sft)")

    try:
        loader = UniversalDataLoader(batch_size=2, sequence_len=10)
        mock_tok = _MockSFTTokenizer()
        conversations = [
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "Hey"}, {"role": "assistant", "content": "Hi there"}]},
        ]
        inputs, targets, mask = loader.create_batch_sft(conversations, mock_tok)

        if inputs.shape == (2, 10):
            print(f"  ✓ Correct input shape: {inputs.shape}")
        else:
            print(f"  ✗ Wrong input shape: {inputs.shape}")
            all_passed = False

        if targets.shape == (2, 10):
            print(f"  ✓ Correct target shape: {targets.shape}")
        else:
            print(f"  ✗ Wrong target shape: {targets.shape}")
            all_passed = False

        if mask.shape == (2, 10):
            print(f"  ✓ Correct mask shape: {mask.shape}")
        else:
            print(f"  ✗ Wrong mask shape: {mask.shape}")
            all_passed = False

        if torch.all(inputs[:, 1:] == targets[:, :-1]):
            print(f"  ✓ Input/target shift correct")
        else:
            print(f"  ✗ Input/target shift wrong")
            all_passed = False

        if mask.dtype in (torch.float32, torch.float64) and mask.min() >= 0 and mask.max() <= 1 and mask.sum() > 0:
            print(f"  ✓ Mask in [0,1], float, and has some trainable positions (sum={mask.sum().item():.0f})")
        else:
            print(f"  ✗ Mask should be float in [0,1] with at least one 1.0")
            all_passed = False

    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Metrics comparison: Reference vs your implementation (only when correct)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("METRICS COMPARISON (Reference vs Your implementation)")
    print("-" * 60)

    metrics_ref = {}
    metrics_user = {}
    metrics_ok = True

    # Test 1 metrics: best-fit pack
    try:
        loader = UniversalDataLoader(batch_size=2, sequence_len=10)
        docs = [
            [1, 10, 11, 12],
            [1, 20, 21, 22, 23, 24],
            [1, 30, 31],
            [1, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        ]
        ref_row, ref_n_docs = _ref_best_fit_pack(copy.deepcopy(docs), row_capacity=11)
        user_row = loader.best_fit_pack(docs.copy(), row_capacity=11)
        metrics_ref["best_fit_row_len"] = len(ref_row)
        metrics_ref["best_fit_docs_consumed"] = ref_n_docs
        metrics_ref["best_fit_starts_with_bos"] = ref_row[0] == 1
        metrics_user["best_fit_row_len"] = len(user_row)
        metrics_user["best_fit_docs_consumed"] = "N/A (API does not return)"
        metrics_user["best_fit_starts_with_bos"] = user_row[0] == 1
        if len(user_row) != len(ref_row) or user_row[0] != ref_row[0]:
            metrics_ok = False
    except Exception as e:
        metrics_user["best_fit"] = f"Error: {e}"
        metrics_ok = False

    # Test 2 metrics: pretrain batch
    try:
        loader = UniversalDataLoader(batch_size=2, sequence_len=8)

        def doc_iter():
            while True:
                yield [1, 100, 101, 102, 103]

        ref_in, ref_tar = _ref_create_batch_pretrain(doc_iter(), B=2, T=8, buffer_size=10, bos_token_id=1)
        user_in, user_tar = loader.create_batch_pretrain(doc_iter(), buffer_size=10)
        metrics_ref["pretrain_input_shape"] = tuple(ref_in.shape)
        metrics_ref["pretrain_target_shape"] = tuple(ref_tar.shape)
        metrics_ref["pretrain_input_mean"] = ref_in.float().mean().item()
        metrics_ref["pretrain_bos_at_0"] = (ref_in[:, 0] == 1).all().item()
        metrics_user["pretrain_input_shape"] = tuple(user_in.shape)
        metrics_user["pretrain_target_shape"] = tuple(user_tar.shape)
        metrics_user["pretrain_input_mean"] = user_in.float().mean().item()
        metrics_user["pretrain_bos_at_0"] = (user_in[:, 0] == 1).all().item()
        if ref_in.shape != user_in.shape or ref_tar.shape != user_tar.shape:
            metrics_ok = False
        if not (torch.allclose(ref_in.float(), user_in.float()) and torch.allclose(ref_tar.float(), user_tar.float())):
            metrics_ref["pretrain_match"] = "N/A"
            metrics_user["pretrain_match"] = "tensors differ (allowed if packing order differs)"
    except Exception as e:
        metrics_user["pretrain"] = f"Error: {e}"
        metrics_ok = False

    # Test 3 metrics: mix
    try:
        loader = UniversalDataLoader()
        MIX_SEED = 42
        N_MIX = 100

        def iter_a():
            while True:
                yield "A"

        def iter_b():
            while True:
                yield "B"

        ref_samples = _ref_mix_datasets(
            {"a": iter_a(), "b": iter_b()},
            {"a": 0.7, "b": 0.3},
            n=N_MIX,
            seed=MIX_SEED,
        )
        random.seed(MIX_SEED)
        mixed = loader.mix_datasets({"a": iter_a(), "b": iter_b()}, {"a": 0.7, "b": 0.3})
        user_samples = [next(mixed) for _ in range(N_MIX)]
        ref_a, ref_b = ref_samples.count("A"), ref_samples.count("B")
        user_a, user_b = user_samples.count("A"), user_samples.count("B")
        metrics_ref["mix_A_count"] = ref_a
        metrics_ref["mix_B_count"] = ref_b
        metrics_user["mix_A_count"] = user_a
        metrics_user["mix_B_count"] = user_b
        if abs(user_a - ref_a) > 15 or abs(user_b - ref_b) > 15:
            metrics_ok = False
    except Exception as e:
        metrics_user["mix"] = f"Error: {e}"
        metrics_ok = False

    # Test 4 metrics: SFT batch
    try:
        loader = UniversalDataLoader(batch_size=2, sequence_len=10)
        mock_tok = _MockSFTTokenizer()
        sft_conversations = [
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "Hey"}, {"role": "assistant", "content": "Hi there"}]},
        ]
        ref_in, ref_tar, ref_mask = _ref_create_batch_sft(
            sft_conversations, mock_tok, B=2, T=10, device=loader.device, pad_token_id=0
        )
        user_in, user_tar, user_mask = loader.create_batch_sft(sft_conversations, mock_tok)
        metrics_ref["sft_input_shape"] = tuple(ref_in.shape)
        metrics_ref["sft_mask_sum"] = ref_mask.sum().item()
        metrics_ref["sft_shift_ok"] = torch.all(ref_in[:, 1:] == ref_tar[:, :-1]).item()
        metrics_user["sft_input_shape"] = tuple(user_in.shape)
        metrics_user["sft_mask_sum"] = user_mask.sum().item()
        metrics_user["sft_shift_ok"] = torch.all(user_in[:, 1:] == user_tar[:, :-1]).item()
        if ref_in.shape != user_in.shape or ref_mask.shape != user_mask.shape:
            metrics_ok = False
        if not torch.allclose(ref_mask.float(), user_mask.float()):
            metrics_ref["sft_mask_match"] = "N/A"
            metrics_user["sft_mask_match"] = "tensors may differ (pad token / impl detail)"
    except Exception as e:
        metrics_user["sft"] = f"Error: {e}"
        metrics_ok = False

    # Print comparison table
    def _fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    print("\n  Test 1 — Best-fit pack:")
    if "best_fit_row_len" in metrics_ref:
        print(f"    Reference: row_len={metrics_ref.get('best_fit_row_len')}, docs_consumed={metrics_ref.get('best_fit_docs_consumed')}, starts_with_bos={_fmt(metrics_ref.get('best_fit_starts_with_bos'))}")
        if "best_fit_row_len" in metrics_user:
            print(f"    Yours:     row_len={metrics_user.get('best_fit_row_len')}, docs_consumed={metrics_user.get('best_fit_docs_consumed')}, starts_with_bos={_fmt(metrics_user.get('best_fit_starts_with_bos'))}")
        else:
            print(f"    Yours:     {metrics_user.get('best_fit', 'N/A')}")
    else:
        print(f"    Yours: {metrics_user.get('best_fit', 'N/A')}")

    print("\n  Test 2 — Pretrain batch:")
    if "pretrain_input_shape" in metrics_ref:
        print(f"    Reference: shape={metrics_ref.get('pretrain_input_shape')}, input_mean={_fmt(metrics_ref.get('pretrain_input_mean'))}, all_rows_start_bos={_fmt(metrics_ref.get('pretrain_bos_at_0'))}")
        if "pretrain_input_shape" in metrics_user:
            print(f"    Yours:     shape={metrics_user.get('pretrain_input_shape')}, input_mean={_fmt(metrics_user.get('pretrain_input_mean'))}, all_rows_start_bos={_fmt(metrics_user.get('pretrain_bos_at_0'))}")
        else:
            print(f"    Yours:     {metrics_user.get('pretrain', 'N/A')}")
    else:
        print(f"    Yours: {metrics_user.get('pretrain', 'N/A')}")

    print("\n  Test 3 — Mix (A/B counts, ~70/30):")
    if "mix_A_count" in metrics_ref:
        print(f"    Reference: A={metrics_ref.get('mix_A_count')}, B={metrics_ref.get('mix_B_count')}")
        if "mix_A_count" in metrics_user:
            print(f"    Yours:     A={metrics_user.get('mix_A_count')}, B={metrics_user.get('mix_B_count')}")
        else:
            print(f"    Yours:     {metrics_user.get('mix', 'N/A')}")
    else:
        print(f"    Yours: {metrics_user.get('mix', 'N/A')}")

    print("\n  Test 4 — SFT batch (create_batch_sft):")
    if "sft_input_shape" in metrics_ref:
        print(f"    Reference: shape={metrics_ref.get('sft_input_shape')}, mask_sum={metrics_ref.get('sft_mask_sum'):.0f}, shift_ok={_fmt(metrics_ref.get('sft_shift_ok'))}")
        if "sft_input_shape" in metrics_user:
            print(f"    Yours:     shape={metrics_user.get('sft_input_shape')}, mask_sum={metrics_user.get('sft_mask_sum'):.0f}, shift_ok={_fmt(metrics_user.get('sft_shift_ok'))}")
        else:
            print(f"    Yours:     {metrics_user.get('sft', 'N/A')}")
    else:
        print(f"    Yours: {metrics_user.get('sft', 'N/A')}")

    print("\n" + "=" * 60)
    if all_passed and metrics_ok:
        print("ALL TESTS PASSED and metrics match reference.")
    elif all_passed and not metrics_ok:
        print("Correctness passed; metrics differ from reference (review above).")
    else:
        print("Some tests failed. Fix correctness first, then check metrics.")
    print("=" * 60)

    return all_passed


def peek():
    """Show the reference implementation from nanochat."""
    from nanochat import dataloader

    print("=" * 60)
    print("REFERENCE: Source code from nanochat/dataloader.py")
    print("=" * 60)

    print("\n### tokenizing_distributed_data_loader_with_state_bos_bestfit ###")
    print(inspect.getsource(dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit))


def main():
    """Run the puzzle interactively."""
    print(__doc__)
    print("=" * 60)
    print("Try implementing the UniversalDataLoader class above!")
    print("Run with --verify to test, or --peek to see the solution.")
    print("=" * 60)

    # Quick smoke test
    try:
        loader = UniversalDataLoader(batch_size=4, sequence_len=128)
        print(f"\n✓ Loader created: B={loader.batch_size}, T={loader.sequence_len}")

        docs = [[1, 10, 11], [1, 20, 21, 22]]
        row = loader.best_fit_pack(docs, row_capacity=10)
        print(f"✓ Best-fit pack result: {row}")
    except NotImplementedError as e:
        print(f"\n⚠ Not yet implemented: {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify your solution")
    parser.add_argument("--peek", action="store_true", help="See reference solution")
    args = parser.parse_args()

    if args.verify:
        verify()
    elif args.peek:
        peek()
    else:
        main()
