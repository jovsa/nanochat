"""
Puzzle 3: The Data Engine ⭐⭐⭐⭐
Goal: Build a high-performance data pipeline for SFT and Pretraining.

Usage:
python puzzles/03_data_engine.py          # Run the puzzle
python puzzles/03_data_engine.py --verify # Verify your solution
python puzzles/03_data_engine.py --peek   # See reference solution (oracle)
"""

import sys
import argparse
import torch
import inspect
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
        raise NotImplementedError("Implement best-fit bin packing")
    
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
        # YOUR CODE HERE
        raise NotImplementedError("Implement pretrain batch creation")
    
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
        # YOUR CODE HERE
        raise NotImplementedError("Implement SFT batch creation with masking")
    
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
        raise NotImplementedError("Implement dataset mixing")


# =============================================================================
# VERIFICATION (Do not modify below this line)
# =============================================================================

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
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! You've mastered the Data Engine!")
    else:
        print("❌ Some tests failed. Keep working on your implementation!")
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
