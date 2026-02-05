"""
Puzzle 5: The RL Loop ⭐⭐⭐⭐⭐
Goal: Build the complete GRPO training loop.

Usage (from repo root):
python -m puzzles.05_rl_loop          # Run the puzzle
python -m puzzles.05_rl_loop --verify # Verify your solution
python -m puzzles.05_rl_loop --peek   # See reference solution (oracle)
"""

import argparse
import torch
import torch.nn.functional as F
import inspect
from typing import Optional

# =============================================================================
# YOUR TASK: Implement the RLTrainer class
# =============================================================================

class RLTrainer:
    """
    Build the complete GRPO (Group Relative Policy Optimization) training loop.
    
    Your implementation should:
    1. Rollout: Generate K samples with temperature=1.0
    2. Advantage: Compute (r - mean) (Simplified) OR (r - mean)/std (Full GRPO)
    3. Loss: Compute token-level policy gradient with masking
    4. KL Penalty: (Optional) Add KL div against a reference model
    
    Hints:
    - Study scripts/chat_rl.py for the training loop structure
    - GRPO advantages are computed per-group (samples from same prompt)
    - Loss is negative log probability weighted by advantage
    """
    
    def __init__(self, model, tokenizer, engine, reward_fn):
        """
        Initialize the RL trainer.
        
        Args:
            model: GPT model to train
            tokenizer: Tokenizer for encoding/decoding
            engine: Inference engine for rollouts
            reward_fn: Function that takes (completion_str) -> float reward
        """
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.reward_fn = reward_fn
    
    def generate_rollouts(self, prompt_tokens: list[int], num_samples: int = 4,
                          max_tokens: int = 256, temperature: float = 1.0
                          ) -> tuple[list[list[int]], torch.Tensor]:
        """
        Generate K rollout samples from the same prompt.
        
        Args:
            prompt_tokens: Tokenized prompt
            num_samples: Number of samples (K) to generate
            max_tokens: Max tokens per sample
            temperature: Sampling temperature (1.0 for exploration)
            
        Returns:
            tuple: (sequences, rewards)
                - sequences: list of K token sequences (each is list of ints)
                - rewards: tensor of shape (K,) with reward for each
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement rollout generation")
    
    def compute_advantages(self, rewards: torch.Tensor, 
                           normalize: bool = True) -> torch.Tensor:
        """
        Compute advantages from rewards (GRPO style).
        
        Args:
            rewards: Tensor of shape (K,) with rewards
            normalize: If True, use (r - mean)/std (Full GRPO)
                      If False, use (r - mean) (Simplified)
        
        Returns:
            Tensor of shape (K,) with advantages
            
        Notes:
        - Subtract mean to center advantages (baseline)
        - Divide by std to normalize scale (reduces variance)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement advantage computation")
    
    def compute_policy_gradient_loss(self, 
                                      sequences: list[list[int]],
                                      advantages: torch.Tensor,
                                      prompt_len: int) -> torch.Tensor:
        """
        Compute token-level policy gradient loss.
        
        Args:
            sequences: List of K token sequences
            advantages: Tensor of shape (K,) with advantage per sequence
            prompt_len: Length of prompt (to mask out prompt tokens)
            
        Returns:
            Scalar loss tensor
            
        Formula:
            L = -sum(advantage_k * sum(log_prob(token_t)))
            
        Only compute loss on generated tokens, not prompt tokens.
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement policy gradient loss")
    
    def train_step(self, prompt_tokens: list[int], 
                   num_samples: int = 4,
                   normalize_advantages: bool = True) -> dict:
        """
        Perform a single GRPO training step.
        
        Args:
            prompt_tokens: Tokenized prompt
            num_samples: Number of rollout samples
            normalize_advantages: Whether to normalize advantages
            
        Returns:
            dict with 'loss', 'mean_reward', 'advantages'
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement training step")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify():
    print("=" * 60)
    print("VERIFICATION: Testing your RLTrainer")
    print("=" * 60)
    
    all_passed = True
    
    print("\nTest 1: Advantage Computation")
    try:
        class MockObjects:
            pass
        trainer = RLTrainer(MockObjects(), MockObjects(), MockObjects(), lambda x: 0.5)
        
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = trainer.compute_advantages(rewards, normalize=False)
        
        # Mean is 2.5, so advantages should be [-1.5, -0.5, 0.5, 1.5]
        expected = rewards - rewards.mean()
        if torch.allclose(adv, expected, atol=0.01):
            print("  ✓ Unnormalized advantages correct")
        else:
            print(f"  ✗ Wrong: {adv}, expected {expected}")
            all_passed = False
            
        adv_norm = trainer.compute_advantages(rewards, normalize=True)
        expected_norm = (rewards - rewards.mean()) / rewards.std()
        if torch.allclose(adv_norm, expected_norm, atol=0.01):
            print("  ✓ Normalized advantages correct")
        else:
            print(f"  ✗ Wrong normalized")
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
    print("=" * 60)
    print("REFERENCE: scripts/chat_rl.py")
    print("=" * 60)
    print("\nKey excerpts from the GRPO implementation:\n")
    print("""
# From scripts/chat_rl.py - get_batch function:
# 1. Generate K samples per example
sequences = engine.generate_batch(tokens, num_samples=num_samples, ...)

# 2. Compute rewards using verifier
rewards = torch.tensor([verify_answer(seq) for seq in sequences])

# 3. Compute advantages (simplified GRPO: just center, no std normalization)
advantages = rewards - rewards.mean()

# 4. Policy gradient loss with masking
logits = model(inputs)
log_probs = F.cross_entropy(logits, targets, reduction='none')
loss = (log_probs * advantages.unsqueeze(-1)).sum()
""")


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
