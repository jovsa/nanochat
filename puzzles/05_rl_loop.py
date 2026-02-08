"""
Puzzle 5: The RL Loop ⭐⭐⭐⭐⭐
Goal: Build the complete GRPO training loop.

Usage (from repo root):
python -m puzzles.05_rl_loop          # Run the puzzle
python -m puzzles.05_rl_loop --verify # Verify your solution
python -m puzzles.05_rl_loop --peek   # See reference solution (oracle)
"""

import argparse
import math
import time
import torch
import torch.nn.functional as F
import inspect
from typing import Optional

# =============================================================================
# YOUR TASK: Implement the RLTrainer class
# =============================================================================


class _MockLogitsModel(torch.nn.Module):
    """Tiny model for tests: forward(idx) returns logits (B, T, V)."""

    def __init__(self, vocab_size: int = 16, device: Optional[torch.device] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self._device = device

    def forward(self, idx: torch.Tensor, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.shape
        device = idx.device
        # Deterministic logits so loss parity test is reproducible
        logits = torch.zeros(B, T, self.vocab_size, device=device, dtype=torch.float32)
        logits[..., 0] = 1.0
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
        return logits


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
        out = self.engine.generate_batch(
            prompt_tokens,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        sequences = out[0]  # list of K token sequences (engine may also return masks)
        prompt_len = len(prompt_tokens)
        rewards_list = []
        for seq in sequences:
            completion_tokens = seq[prompt_len:]
            completion_str = self.tokenizer.decode(completion_tokens)
            rewards_list.append(self.reward_fn(completion_str))
        rewards = torch.tensor(rewards_list, dtype=torch.float)
        return sequences, rewards

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
        centered = rewards - rewards.mean()
        if normalize:
            std = rewards.std()
            if std > 1e-8:
                centered = centered / std
        return centered

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
            Tuple of (scalar loss tensor, num_valid_tokens int).

        Formula:
            L = -sum(advantage_k * sum(log_prob(token_t)))

        Only compute loss on generated tokens, not prompt tokens.
        """
        device = next(self.model.parameters()).device
        advantages = advantages.to(device)
        max_len = max(len(s) for s in sequences)
        padded = [s + [0] * (max_len - len(s)) for s in sequences]
        ids = torch.tensor(padded, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[:, : prompt_len - 1] = -1
        for b, seq in enumerate(sequences):
            if len(seq) < max_len:
                targets[b, len(seq) - 1 :] = -1
        num_valid_tokens = (targets >= 0).sum().item()
        logits = self.model(inputs)
        # logits (B, T, V), targets (B, T) -> flatten to (B*T, V) and (B*T,)
        log_probs = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
            ignore_index=-1,
        )
        log_probs = log_probs.view(targets.shape)
        loss = (log_probs * advantages.unsqueeze(-1)).sum()
        return loss, num_valid_tokens

    def train_step(self, prompt_tokens: list[int],
                   num_samples: int = 4,
                   normalize_advantages: bool = True,
                   record_timings: bool = False,
                   record_memory: bool = False) -> dict:
        """
        Perform a single GRPO training step.

        Args:
            prompt_tokens: Tokenized prompt
            num_samples: Number of rollout samples
            normalize_advantages: Whether to normalize advantages
            record_timings: If True, record rollout/loss/step times (CUDA sync when available)
            record_memory: If True and CUDA available, record peak GPU memory in MiB

        Returns:
            dict with 'loss', 'mean_reward', 'advantages', training stats, optional bpb,
            and when recording: timing, memory, throughput metrics.
        """
        cuda_available = torch.cuda.is_available()
        if record_memory and not cuda_available:
            record_memory = False

        if record_timings and cuda_available:
            torch.cuda.synchronize()
        step_t0 = time.perf_counter()

        if record_memory and cuda_available:
            torch.cuda.reset_peak_memory_stats()

        # Rollout
        if record_timings and cuda_available:
            torch.cuda.synchronize()
        rollout_t0 = time.perf_counter()
        sequences, rewards = self.generate_rollouts(
            prompt_tokens, num_samples=num_samples
        )
        if record_timings and cuda_available:
            torch.cuda.synchronize()
        rollout_t1 = time.perf_counter()
        rollout_time_s = rollout_t1 - rollout_t0 if record_timings else None

        if record_memory and cuda_available:
            memory_after_rollout_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        advantages = self.compute_advantages(
            rewards, normalize=normalize_advantages
        )
        prompt_len = len(prompt_tokens)

        # Loss
        if record_timings and cuda_available:
            torch.cuda.synchronize()
        loss_t0 = time.perf_counter()
        loss, num_valid_tokens = self.compute_policy_gradient_loss(
            sequences, advantages, prompt_len
        )
        if record_timings and cuda_available:
            torch.cuda.synchronize()
        loss_t1 = time.perf_counter()
        loss_time_s = loss_t1 - loss_t0 if record_timings else None

        if record_timings and cuda_available:
            torch.cuda.synchronize()
        step_t1 = time.perf_counter()
        step_time_s = step_t1 - step_t0 if record_timings else None

        # Build result
        mean_reward = rewards.float().mean()
        reward_std = rewards.float().std()
        advantages_std = advantages.std()
        mean_sequence_length = sum(len(s) for s in sequences) / len(sequences) if sequences else 0.0

        bpb = None
        if num_valid_tokens > 0:
            bpb = (loss.item() / num_valid_tokens) * math.log2(math.e)

        out = {
            "loss": loss,
            "mean_reward": mean_reward,
            "advantages": advantages,
            "reward_std": reward_std,
            "advantages_std": advantages_std,
            "mean_sequence_length": mean_sequence_length,
            "num_valid_tokens": num_valid_tokens,
            "bpb": bpb,
        }

        if record_timings:
            out["rollout_time_s"] = rollout_time_s
            out["loss_time_s"] = loss_time_s
            out["step_time_s"] = step_time_s
            num_samples_actual = len(sequences)
            total_gen_tokens = sum(max(0, len(s) - prompt_len) for s in sequences)
            out["samples_per_sec"] = num_samples_actual / rollout_time_s if rollout_time_s and rollout_time_s > 0 else None
            out["gen_tokens_per_sec"] = total_gen_tokens / rollout_time_s if rollout_time_s and rollout_time_s > 0 else None
            out["loss_tokens_per_sec"] = num_valid_tokens / step_time_s if step_time_s and step_time_s > 0 else None

        if record_memory and cuda_available:
            out["peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            out["memory_after_rollout_mb"] = memory_after_rollout_mb

        return out


# =============================================================================
# VERIFICATION
# =============================================================================

def verify():
    print("=" * 60)
    print("VERIFICATION: Testing your RLTrainer")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("CUDA not available; skipping GPU-dependent tests and metrics.")
    else:
        print("CUDA available; running full verification including timing/memory.")

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

    print("\nTest 2: Loss formula parity (vs chat_rl-style manual)")
    try:
        device = torch.device("cuda" if cuda_available else "cpu")
        vocab_size = 16
        mock_model = _MockLogitsModel(vocab_size=vocab_size).to(device)
        class MockObjects2:
            pass
        mock_tok = MockObjects2()
        mock_engine = MockObjects2()
        trainer = RLTrainer(mock_model, mock_tok, mock_engine, lambda x: 0.0)

        prompt_len = 2
        sequences = [
            [1, 2, 3, 4, 5],
            [1, 2, 6, 7, 8],
            [1, 2, 9, 10, 11],
        ]
        advantages = torch.tensor([0.5, -0.3, 0.2], device=device, dtype=torch.float32)

        loss_puzzle, num_valid = trainer.compute_policy_gradient_loss(
            sequences, advantages, prompt_len
        )

        max_len = max(len(s) for s in sequences)
        padded = [s + [0] * (max_len - len(s)) for s in sequences]
        ids = torch.tensor(padded, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[:, : prompt_len - 1] = -1
        for b, seq in enumerate(sequences):
            if len(seq) < max_len:
                targets[b, len(seq) - 1 :] = -1

        logits = mock_model(inputs)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
            ignore_index=-1,
        )
        nll = nll.view(targets.shape)
        loss_manual = (nll * advantages.unsqueeze(-1)).sum()

        if torch.allclose(loss_puzzle, loss_manual, atol=1e-5):
            print("  ✓ Loss formula matches chat_rl-style manual computation")
        else:
            print(f"  ✗ Loss mismatch: puzzle={loss_puzzle.item()}, manual={loss_manual.item()}")
            all_passed = False

    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\nDemo: one train_step with mock data and metrics")
    try:
        class MockEngine:
            def generate_batch(self, prompt_tokens, num_samples=1, **kwargs):
                # Different completion lengths so rewards/advantages vary and loss != 0
                completions = [[1, 2, 3], [1, 2, 3, 4]][:num_samples]
                out_seqs = [list(prompt_tokens) + c for c in completions]
                masks = [[0] * len(prompt_tokens) + [1] * len(c) for c in completions]
                return out_seqs, masks

        class MockTokenizer:
            def decode(self, tokens):
                return "x" * len(tokens)  # length varies per completion for demo_reward

        device = torch.device("cuda" if cuda_available else "cpu")
        mock_model = _MockLogitsModel(vocab_size=16).to(device)
        mock_engine = MockEngine()
        mock_tok = MockTokenizer()
        # Use varying rewards so advantages are non-zero and loss is non-zero
        def demo_reward(completion):
            return 0.3 + 0.1 * len(completion)  # vary by completion length
        trainer = RLTrainer(mock_model, mock_tok, mock_engine, demo_reward)

        prompt_tokens = [1, 2, 3]
        result = trainer.train_step(
            prompt_tokens,
            num_samples=2,
            record_timings=cuda_available,
            record_memory=cuda_available,
        )

        print("  Metrics:")
        for k, v in result.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                print(f"    {k}: {v.item():.6f}")
            elif isinstance(v, (int, float)):
                print(f"    {k}: {v}")
            elif v is None:
                print(f"    {k}: None")
            elif k == "advantages":
                print(f"    {k}: tensor(shape {v.shape})")
            else:
                print(f"    {k}: {v}")

    except Exception as e:
        print(f"  ✗ Demo error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    print("PASSED!" if all_passed else "Failed")
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


def run_benchmark(num_steps: int = 3):
    """Run N train steps with timing/memory/throughput and print summary."""
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("CUDA not available; benchmark will run without GPU timing/memory.")

    class MockEngine:
        def generate_batch(self, prompt_tokens, num_samples=1, **kwargs):
            out_seqs = [list(prompt_tokens) + [1, 2, 3, 4] for _ in range(num_samples)]
            masks = [[0] * len(prompt_tokens) + [1] * 4 for _ in range(num_samples)]
            return out_seqs, masks

    class MockTokenizer:
        def decode(self, tokens):
            return ""

    device = torch.device("cuda" if cuda_available else "cpu")
    mock_model = _MockLogitsModel(vocab_size=16).to(device)
    mock_engine = MockEngine()
    mock_tok = MockTokenizer()
    trainer = RLTrainer(mock_model, mock_tok, mock_engine, lambda x: 0.5)

    prompt_tokens = [1, 2, 3]
    results = []
    for _ in range(num_steps):
        result = trainer.train_step(
            prompt_tokens,
            num_samples=2,
            record_timings=True,
            record_memory=cuda_available,
        )
        results.append(result)

    # Summary
    losses = [r["loss"].item() for r in results]
    bpbs = [r["bpb"] for r in results if r.get("bpb") is not None]
    print("\nBenchmark summary:")
    print(f"  steps: {num_steps}")
    print(f"  mean loss: {sum(losses) / len(losses):.6f}")
    if bpbs:
        print(f"  mean bpb: {sum(bpbs) / len(bpbs):.6f}")
    if results and results[0].get("step_time_s") is not None:
        step_times = [r["step_time_s"] for r in results if r.get("step_time_s") is not None]
        print(f"  mean step_time_s: {sum(step_times) / len(step_times):.4f}")
    if results and results[0].get("peak_memory_allocated_mb") is not None:
        mems = [r["peak_memory_allocated_mb"] for r in results if r.get("peak_memory_allocated_mb") is not None]
        print(f"  mean peak_memory_mb: {sum(mems) / len(mems):.2f}")
    if results and results[0].get("gen_tokens_per_sec") is not None:
        tps = [r["gen_tokens_per_sec"] for r in results if r.get("gen_tokens_per_sec") is not None]
        print(f"  mean gen_tokens_per_sec: {sum(tps) / len(tps):.2f}")


def main():
    print(__doc__)
    print("Run with --verify to test, --peek to see the solution, or --benchmark for timing/memory summary.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--peek", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Run N steps with timing/memory/throughput; print summary")
    parser.add_argument("--benchmark-steps", type=int, default=3, help="Number of steps for --benchmark")
    args = parser.parse_args()

    if args.verify:
        verify()
    elif args.peek:
        peek()
    elif args.benchmark:
        run_benchmark(num_steps=args.benchmark_steps)
    else:
        main()
