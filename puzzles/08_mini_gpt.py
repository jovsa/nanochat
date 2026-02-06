"""
Puzzle 8: Mini GPT from Scratch ⭐⭐⭐⭐⭐
Goal: Build the model itself.

Usage (from repo root):
python -m puzzles.08_mini_gpt          # Run the puzzle
python -m puzzles.08_mini_gpt --verify # Verify your solution
python -m puzzles.08_mini_gpt --peek   # See reference solution (oracle)
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from dataclasses import dataclass

# =============================================================================
# YOUR TASK: Implement the MiniGPT model
# =============================================================================

@dataclass
class MiniGPTConfig:
    """Configuration for MiniGPT."""
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6  # For GQA
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x: torch.Tensor) -> torch.Tensor:
    """
    RMSNorm without learnable parameters.

    Formula: x / sqrt(mean(x^2) + eps)

    Hint: Use F.rms_norm
    """
    # YOUR CODE HERE
    eps = 1e-15
    mean_sq = torch.mean(x ** 2, dim=-1, keepdim=True)
    norm = x / torch.sqrt(mean_sq + eps)
    return norm


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings (RoPE).

    Args:
        x: Input tensor of shape (B, T, H, D)
        cos, sin: Rotation frequencies of shape (1, T, 1, D//2)

    Returns:
        Rotated tensor of same shape

    Formula:
        Split x into two halves (x1, x2)
        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos
        y = concat(y1, y2)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement RoPE")


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention with GQA support.

    Features:
    - Group Query Attention (n_kv_head <= n_head)
    - RoPE positional encoding
    - QK normalization
    - Optional sliding window
    """

    def __init__(self, config: MiniGPTConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        # YOUR CODE HERE: Define linear layers
        # c_q: n_embd -> n_head * head_dim
        # c_k: n_embd -> n_kv_head * head_dim
        # c_v: n_embd -> n_kv_head * head_dim
        # c_proj: n_embd -> n_embd
        raise NotImplementedError("Implement attention layers")

    def forward(self, x: torch.Tensor, cos_sin: tuple,
                window_size: tuple = (-1, 0)) -> torch.Tensor:
        """
        Forward pass for attention.

        Args:
            x: Input of shape (B, T, C)
            cos_sin: Tuple of (cos, sin) for RoPE
            window_size: (left, right) for sliding window, -1 = unlimited

        Returns:
            Output of shape (B, T, C)
        """
        # YOUR CODE HERE
        # 1. Project to Q, K, V
        # 2. Reshape to (B, T, H, D)
        # 3. Apply RoPE to Q and K
        # 4. Apply QK normalization
        # 5. Compute attention (can use F.scaled_dot_product_attention)
        # 6. Project output
        raise NotImplementedError("Implement attention forward")


class MLP(nn.Module):
    """
    MLP with ReLU^2 activation.

    Structure: x -> fc -> relu^2 -> proj -> out
    Expansion factor: 4x
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError("Implement MLP layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP with ReLU^2 activation."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement MLP forward")


class Block(nn.Module):
    """Transformer block: Attention + MLP with residual connections."""

    def __init__(self, config: MiniGPTConfig, layer_idx: int):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError("Implement block")

    def forward(self, x: torch.Tensor, cos_sin: tuple,
                window_size: tuple) -> torch.Tensor:
        """Block forward with pre-norm and residuals."""
        # YOUR CODE HERE
        # x = x + attn(norm(x))
        # x = x + mlp(norm(x))
        raise NotImplementedError("Implement block forward")


class MiniGPT(nn.Module):
    """
    Complete GPT model.

    Architecture:
    - Token embedding (untied from lm_head)
    - Stack of transformer blocks
    - RMSNorm before lm_head
    - Logit softcapping at 15
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        # YOUR CODE HERE: Build the model
        # - wte: Embedding
        # - blocks: ModuleList of Block
        # - lm_head: Linear
        # - Register cos/sin buffers for RoPE
        raise NotImplementedError("Implement model init")

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            idx: Token IDs of shape (B, T)
            targets: Optional target IDs for loss computation

        Returns:
            If targets: scalar loss
            Else: logits of shape (B, T, vocab_size)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement forward pass")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify():
    print("=" * 60)
    print("VERIFICATION: Testing your MiniGPT")
    print("=" * 60)

    all_passed = True

    print("\nTest 1: RMSNorm")
    try:
        x = torch.randn(2, 4, 8)
        y = norm(x)

        # Check shape preserved
        if y.shape == x.shape:
            print("  ✓ Shape preserved")
        else:
            print(f"  ✗ Wrong shape: {y.shape}")
            all_passed = False

        # Check normalized (RMS should be ~1)
        rms = (y ** 2).mean(dim=-1).sqrt()
        if torch.allclose(rms, torch.ones_like(rms), atol=0.1):
            print("  ✓ RMS is ~1")
        else:
            print(f"  ✗ RMS not ~1: {rms.mean()}")
            all_passed = False

    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False

    print("\nTest 2: RoPE")
    try:
        x = torch.randn(1, 4, 2, 8)  # B, T, H, D
        cos = torch.ones(1, 4, 1, 4)  # D//2
        sin = torch.zeros(1, 4, 1, 4)

        y = apply_rotary_emb(x, cos, sin)

        # With cos=1, sin=0, output should equal input
        if torch.allclose(y, x, atol=0.01):
            print("  ✓ Identity rotation works")
        else:
            print("  ✗ Identity rotation failed")
            all_passed = False

    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False

    print("\nTest 3: MLP")
    try:
        config = MiniGPTConfig(n_embd=64)
        mlp = MLP(config)

        x = torch.randn(2, 4, 64)
        y = mlp(x)

        if y.shape == (2, 4, 64):
            print("  ✓ MLP shape correct")
        else:
            print(f"  ✗ Wrong shape: {y.shape}")
            all_passed = False

    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False

    print("\nTest 4: Full Model")
    try:
        config = MiniGPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=64, vocab_size=256)
        model = MiniGPT(config)

        idx = torch.randint(0, 256, (2, 16))
        logits = model(idx)

        if logits.shape == (2, 16, 256):
            print("  ✓ Output shape correct")
        else:
            print(f"  ✗ Wrong shape: {logits.shape}")
            all_passed = False

        # Check softcap
        if logits.max() <= 15.01 and logits.min() >= -15.01:
            print("  ✓ Logit softcapping works")
        else:
            print(f"  ✗ Logits not capped: [{logits.min()}, {logits.max()}]")
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
    from nanochat import gpt
    print("=" * 60)
    print("REFERENCE: nanochat/gpt.py")
    print("=" * 60)
    print("\n### norm ###")
    print(inspect.getsource(gpt.norm))
    print("\n### apply_rotary_emb ###")
    print(inspect.getsource(gpt.apply_rotary_emb))
    print("\n### MLP ###")
    print(inspect.getsource(gpt.MLP))
    print("\n### CausalSelfAttention ###")
    print(inspect.getsource(gpt.CausalSelfAttention))


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
