"""
Puzzle 1: The Transformer Blueprint ⭐⭐⭐⭐
Goal: Build the complete static graph setup for GPT.

Usage (from repo root):
python -m puzzles.01_transformer_blueprint          # Run the puzzle
python -m puzzles.01_transformer_blueprint --verify # Verify your solution
python -m puzzles.01_transformer_blueprint --peek   # See reference solution (oracle)
"""

import sys
import argparse
import torch
import inspect

# =============================================================================
# YOUR TASK: Implement the ModelSetup class
# =============================================================================

class ModelSetup:
    """
    Build the complete static graph setup for GPT.

    Your implementation should:
    1. Derive GPTConfig from depth (scaling laws logic)
    2. Compute RoPE frequencies (cos, sin) with proper dtype handling
    3. Generate the Sliding Window Attention mask for any pattern (e.g., "SSSL")

    Hints:
    - Study nanochat/gpt.py: GPTConfig, _precompute_rotary_embeddings, _compute_window_sizes
    - The scaling from depth to config follows specific formulas
    - RoPE uses base theta of 10000 by default
    - Window pattern is tiled across layers, final layer always gets full context
    """

    def __init__(self, depth: int, sequence_len: int = 2048, window_pattern: str = "SSSL"):
        """
        Initialize ModelSetup with configuration derived from depth.

        Args:
            depth: Number of transformer layers (n_layer)
            sequence_len: Maximum sequence length
            window_pattern: Sliding window attention pattern string (e.g., "SSSL")
        """
        # TODO: Derive configuration parameters from depth
        # In nanochat, the model uses specific ratios:
        # - n_head = n_layer // 2 (minimum 1)
        # - n_kv_head = n_head (GQA disabled by default)
        # - n_embd = n_head * 128 (head_dim = 128)
        # - vocab_size = 32768 (default)

        self.n_layer = depth
        self.sequence_len = sequence_len
        self.window_pattern = window_pattern

        # Derive config from depth (same as nanochat GPTConfig)
        self.n_head = max(1, depth // 2)
        self.n_kv_head = self.n_head
        self.n_embd = self.n_head * 128
        self.vocab_size = 32768


    def compute_rotary_embeddings(self, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE (Rotary Position Embeddings) frequencies.

        Returns:
            tuple: (cos, sin) tensors of shape (1, seq_len, 1, head_dim//2) in bfloat16

        Hints:
        - head_dim = n_embd // n_head
        - Use base theta = 10000
        - inv_freq = 1 / (base ** (channel_range / head_dim))
        - freqs = outer(t, inv_freq)
        - Final shapes need batch and head dims added: (1, T, 1, head_dim//2)
        """

        base = 10000
        head_dim = self.n_embd // self.n_head

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)

        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(self.sequence_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin


    def compute_window_sizes(self) -> list[tuple[int, int]]:
        """
        Compute per-layer window sizes for sliding window attention.

        Returns:
            List of (left, right) tuples for each layer.
            - left: tokens before current position to attend to
            - right: always 0 for causal attention

        Pattern characters:
            - 'L' = long (full context = sequence_len)
            - 'S' = short (half context = sequence_len // 2)

        Rules:
            - Pattern is tiled across layers
            - Final layer ALWAYS gets full context regardless of pattern
        """
        long_window = self.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L" : (long_window, 0),
            "S" : (short_window, 0),
        }

        window_sizes = []
        for layer_idx in range(self.n_layer):
            char = self.window_pattern[layer_idx % len(self.window_pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes




# =============================================================================
# VERIFICATION (Do not modify below this line)
# =============================================================================

def verify():
    """Verify your implementation against the nanochat ground truth."""
    from nanochat.gpt import GPT, GPTConfig

    print("=" * 60)
    print("VERIFICATION: Testing your ModelSetup against nanochat.gpt")
    print("=" * 60)

    test_cases = [
        {"depth": 12, "sequence_len": 2048, "window_pattern": "SSSL"},
        {"depth": 6, "sequence_len": 1024, "window_pattern": "SL"},
        {"depth": 24, "sequence_len": 4096, "window_pattern": "L"},
    ]

    all_passed = True

    for i, tc in enumerate(test_cases):
        print(f"\nTest Case {i+1}: depth={tc['depth']}, seq_len={tc['sequence_len']}, pattern='{tc['window_pattern']}'")

        try:
            # Your implementation
            setup = ModelSetup(**tc)
            your_cos, your_sin = setup.compute_rotary_embeddings()
            your_windows = setup.compute_window_sizes()

            # Ground truth from nanochat
            config = GPTConfig(
                n_layer=tc["depth"],
                sequence_len=tc["sequence_len"],
                window_pattern=tc["window_pattern"],
                n_head=max(1, tc["depth"] // 2),
                n_kv_head=max(1, tc["depth"] // 2),
                n_embd=max(1, tc["depth"] // 2) * 128,
            )

            with torch.device("meta"):
                model = GPT(config)

            # Compare window sizes
            if your_windows == model.window_sizes:
                print(f"  ✓ Window sizes match")
            else:
                print(f"  ✗ Window sizes mismatch")
                print(f"    Expected: {model.window_sizes}")
                print(f"    Got: {your_windows}")
                all_passed = False

            # Compare config values
            if (setup.n_layer == config.n_layer and
                setup.n_head == config.n_head and
                setup.n_embd == config.n_embd):
                print(f"  ✓ Config values match")
            else:
                print(f"  ✗ Config mismatch")
                all_passed = False

        except NotImplementedError as e:
            print(f"  ✗ Not implemented: {e}")
            all_passed = False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! You've mastered the Transformer Blueprint!")
    else:
        print("❌ Some tests failed. Keep working on your implementation!")
    print("=" * 60)

    return all_passed


def peek():
    """Show the reference implementation from nanochat."""
    from nanochat import gpt

    print("=" * 60)
    print("REFERENCE: Source code from nanochat/gpt.py")
    print("=" * 60)

    # Show GPTConfig
    print("\n### GPTConfig ###")
    print(inspect.getsource(gpt.GPTConfig))

    # Show _precompute_rotary_embeddings
    print("\n### GPT._precompute_rotary_embeddings ###")
    print(inspect.getsource(gpt.GPT._precompute_rotary_embeddings))

    # Show _compute_window_sizes
    print("\n### GPT._compute_window_sizes ###")
    print(inspect.getsource(gpt.GPT._compute_window_sizes))


def main():
    """Run the puzzle interactively."""
    print(__doc__)
    print("=" * 60)
    print("Try implementing the ModelSetup class above!")
    print("Run with --verify to test, or --peek to see the solution.")
    print("=" * 60)

    # Quick smoke test
    try:
        setup = ModelSetup(depth=12)
        print(f"\n✓ ModelSetup instantiated successfully!")
        print(f"  n_layer: {setup.n_layer}")
        print(f"  n_head: {setup.n_head}")
        print(f"  n_embd: {setup.n_embd}")

        cos, sin = setup.compute_rotary_embeddings()
        print(f"  cos shape: {cos.shape}")
        print(f"  sin shape: {sin.shape}")

        windows = setup.compute_window_sizes()
        print(f"  window_sizes: {windows}")
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
