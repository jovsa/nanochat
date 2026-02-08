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
import importlib

_transformer_blueprint = importlib.import_module("puzzles.01_transformer_blueprint")
ModelSetup = _transformer_blueprint.ModelSetup

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
    assert x.ndim == 4
    d = x.shape[3]//2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)



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


        # c_q: n_embd -> n_head * head_dim
        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        # c_k: n_embd -> n_kv_head * head_dim
        self.c_k = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        # c_v: n_embd -> n_kv_head * head_dim
        self.c_v = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)

        # c_proj: n_embd -> n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

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
        B, T, C = x.size()

        # 1. Project to Q, K, V and reshape to (B, T, H, D)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)   # (B, T, H, D)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)   # (B, T, H, D)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)   # (B, T, H, D)

        # 2. Apply RoPE to Q and K; cos/sin (1, T, 1, D/2) broadcast with (B, T, H, D)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)  # (B, T, H, D)

        # 3. Apply QK normalization
        q, k = norm(q), norm(k)  # (B, T, H, D)

        # 4. Transpose to (B, H, T, D) for attention matmul
        q = q.transpose(1, 2)   # (B, H, T, D)
        k = k.transpose(1, 2)   # (B, H, T, D)
        v = v.transpose(1, 2)   # (B, H, T, D)

        # 5. Compute attention: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)

        # 5.1 Apply causal mask (1, 1, T, T) broadcasts to (B, H, T, T)
        causal_mask = torch.tril(torch.ones(T, T, device=scores.device)).view(1, 1, T, T)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))  # (B, H, T, T)

        # 5.2 Get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)

        out = attn_weights @ v  # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)

        # 6. Project output: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.c_proj(out)  # (B, T, C)
        return out



class MLP(nn.Module):
    """
    MLP with ReLU^2 activation.

    Structure: x -> fc -> relu^2 -> proj -> out
    Expansion factor: 4x
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP with ReLU^2 activation."""
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x



class Block(nn.Module):
    """Transformer block: Attention + MLP with residual connections."""

    def __init__(self, config: MiniGPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cos_sin: tuple,
                window_size: tuple) -> torch.Tensor:
        """Block forward with pre-norm and residuals."""
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


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
        # Token embedding
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        })

        # Language model head (untied from embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precompute RoPE embeddings
        head_dim = config.n_embd // config.n_head

        setup = ModelSetup(
            depth=config.n_layer,
            sequence_len=config.sequence_len,
            window_pattern=config.window_pattern)
        # Puzzle 1 derives n_embd/n_head from depth; we need to match our config
        setup.n_embd = config.n_embd
        setup.n_head = config.n_head

        cos, sin = setup.compute_rotary_embeddings()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Compute window sizes for each layer based on pattern
        self.window_sizes = setup.compute_window_sizes()


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
        B, T = idx.size()

        # Get rotary embeddings for current sequence length
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # 1. Embed tokens
        x = self.transformer.wte(idx)  # (B, T, n_embd)

        # 2. Pass through transformer blocks
        for i, block in enumerate(self.transformer.h):
            x = block(x, cos_sin, self.window_sizes[i])

        # 3. Final normalization
        x = norm(x)

        # 4. Project to vocab (lm_head)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 5. Apply softcapping at 15
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)

        # 6. Compute loss if targets provided
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),                   # (B*T,)
                reduction='mean'
            )
            return loss
        else:
            return logits


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
