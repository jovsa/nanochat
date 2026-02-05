"""
Puzzle 4: The Inference Engine ⭐⭐⭐⭐⭐
Goal: Build the streaming generation loop with tool use.

Usage (from repo root):
python -m puzzles.04_inference_engine          # Run the puzzle
python -m puzzles.04_inference_engine --verify # Verify your solution
python -m puzzles.04_inference_engine --peek   # See reference solution (oracle)
"""

import argparse
import torch
import torch.nn.functional as F
import inspect
from collections import deque
from typing import Optional, Generator
from dataclasses import dataclass, field

# =============================================================================
# YOUR TASK: Implement the KVCache and Engine classes
# =============================================================================

@dataclass
class RowState:
    """State for a single generation row (for tool use handling)."""
    current_tokens: list = field(default_factory=list)
    forced_tokens: deque = field(default_factory=deque)
    in_python_block: bool = False
    python_expr_tokens: list = field(default_factory=list)
    completed: bool = False


class KVCache:
    """
    KV Cache for Flash Attention 3's flash_attn_with_kvcache API.
    Layout: (B, T, H, D). Track cache_seqlens for each batch element.
    
    Hints: Study nanochat/engine.py: KVCache class
    """
    
    def __init__(self, batch_size: int, num_heads: int, seq_len: int, 
                 head_dim: int, num_layers: int, device: str, dtype: torch.dtype):
        # YOUR CODE HERE
        raise NotImplementedError("Implement KV cache initialization")
    
    def reset(self):
        """Reset cache to empty state."""
        raise NotImplementedError("Implement cache reset")
    
    def get_pos(self) -> int:
        """Get current position."""
        raise NotImplementedError("Implement get_pos")
    
    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k_cache, v_cache) for a specific layer."""
        raise NotImplementedError("Implement get_layer_cache")
    
    def advance(self, num_tokens: int):
        """Advance the cache position."""
        raise NotImplementedError("Implement cache advance")
    
    def prefill(self, other: 'KVCache'):
        """Copy cached KV from another cache."""
        raise NotImplementedError("Implement cache prefill")


class Engine:
    """
    Inference engine with streaming generation and tool use support.
    Hints: Study nanochat/engine.py: Engine class
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def sample_next_token(self, logits: torch.Tensor, temperature: float = 1.0,
                          top_k: Optional[int] = None,
                          rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample next token from logits (B, vocab_size) -> (B, 1)."""
        raise NotImplementedError("Implement token sampling")
    
    def generate(self, tokens: list[int], max_tokens: int = 256,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 seed: int = 42, use_tools: bool = True) -> Generator[int, None, None]:
        """Generate tokens with optional tool use."""
        raise NotImplementedError("Implement generation")
    
    def generate_batch(self, tokens: list[int], num_samples: int = 1, **kwargs) -> list[list[int]]:
        """Generate multiple samples from the same prompt."""
        raise NotImplementedError("Implement batch generation")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify():
    print("=" * 60)
    print("VERIFICATION: Testing your Inference Engine")
    print("=" * 60)
    
    all_passed = True
    
    print("\nTest 1: KV Cache Operations")
    try:
        cache = KVCache(2, 4, 128, 64, 6, "cpu", torch.float32)
        if cache.get_pos() == 0:
            print("  ✓ Initial position is 0")
        else:
            print(f"  ✗ Initial position wrong")
            all_passed = False
        cache.advance(10)
        if cache.get_pos() == 10:
            print("  ✓ Position advanced correctly")
        else:
            all_passed = False
        k, v = cache.get_layer_cache(0)
        if k.shape == (2, 128, 4, 64):
            print("  ✓ Layer cache shapes correct")
        else:
            all_passed = False
    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    print("🎉 PASSED!" if all_passed else "❌ Failed")
    return all_passed


def peek():
    from nanochat import engine
    print("=" * 60)
    print("REFERENCE: nanochat/engine.py")
    print("=" * 60)
    print("\n### KVCache ###")
    print(inspect.getsource(engine.KVCache))
    print("\n### Engine.generate ###")
    print(inspect.getsource(engine.Engine.generate))


def main():
    print(__doc__)
    print("Run with --verify to test, or --peek to see the solution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--peek", action="store_true")
    args = parser.parse_args()
    
    if args.verify: verify()
    elif args.peek: peek()
    else: main()
