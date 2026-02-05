"""
Puzzle 2: The Optimization Layer ⭐⭐⭐⭐⭐
Goal: Build the training stepper with Muon optimizer and LR schedules.

Usage:
python puzzles/02_optimization_layer.py          # Run the puzzle
python puzzles/02_optimization_layer.py --verify # Verify your solution
python puzzles/02_optimization_layer.py --peek   # See reference solution (oracle)
"""

import sys
import argparse
import torch
import torch.nn as nn
import inspect

# =============================================================================
# YOUR TASK: Implement the OptimizerFactory class
# =============================================================================

class OptimizerFactory:
    """
    Build the training stepper combining Muon and AdamW optimizers.
    
    Your implementation should:
    1. Split params into 3 groups: muon_matrix (2D), adam_decay, adam_no_decay
    2. Implement the Newton-Schulz/Polar Express update step (Muon core)
    3. Derive the warm-up/warm-down LR schedule
    
    Hints:
    - Study nanochat/optim.py: MuonAdamW, muon_step_fused, adamw_step_fused
    - Muon is for 2D matrix parameters (weight matrices)
    - AdamW is for embeddings, biases, scalars, 1D params
    - Newton-Schulz iteration orthogonalizes the gradient
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the optimizer factory with a model.
        
        Args:
            model: The neural network model to optimize
        """
        self.model = model
        
    def split_param_groups(self) -> dict:
        """
        Split model parameters into optimizer groups.
        
        Returns:
            dict with keys:
                'muon_matrix': list of 2D weight tensors (for Muon)
                'adam_decay': list of params that get weight decay (usually none in nanochat)
                'adam_no_decay': list of params without weight decay (embeddings, biases, 1D)
        
        Rules:
        - 2D params (matrices) -> Muon
        - 1D params, embeddings, scalars -> AdamW (no decay)
        - In nanochat, weight_decay is typically 0, so adam_decay is often empty
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement parameter group splitting")
    
    def newton_schulz_step(self, G: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
        """
        Perform Newton-Schulz iteration to orthogonalize the gradient.
        
        This is the core of Muon optimizer - it computes the polar decomposition
        of the gradient matrix to get an orthogonal update direction.
        
        Args:
            G: Gradient matrix of shape (out_features, in_features)
            num_iters: Number of Newton-Schulz iterations (default 5)
            
        Returns:
            Orthogonalized gradient matrix of same shape
            
        Hints:
        - Newton-Schulz iteration: X_{k+1} = X_k @ (aI + bX_k^TX_k + cX_k^TX_kX_k^TX_k)
        - Or use Polar Express which is faster (see polar_express_coeffs in optim.py)
        - The iteration converges to Q in the polar decomposition G = QP
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement Newton-Schulz orthogonalization")
    
    def get_lr_schedule(self, warmup_steps: int, total_steps: int, 
                        init_lr_frac: float = 0.1) -> callable:
        """
        Create a learning rate schedule with warmup and warmdown.
        
        Args:
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            init_lr_frac: Initial LR as fraction of base LR
            
        Returns:
            Function that takes step number and returns LR multiplier
            
        Schedule:
        - Warmup: linear from init_lr_frac to 1.0 over warmup_steps
        - Warmdown: linear from 1.0 to 0.0 from warmup_steps to total_steps
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement LR schedule")


# =============================================================================
# VERIFICATION (Do not modify below this line)
# =============================================================================

def verify():
    """Verify your implementation against the nanochat ground truth."""
    from nanochat.optim import MuonAdamW, polar_express_coeffs
    
    print("=" * 60)
    print("VERIFICATION: Testing your OptimizerFactory")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Parameter group splitting
    print("\nTest 1: Parameter Group Splitting")
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 256)  # 2D but embedding
            self.linear = nn.Linear(256, 512, bias=False)  # 2D matrix
            self.proj = nn.Linear(512, 256, bias=True)  # 2D matrix + 1D bias
            self.scale = nn.Parameter(torch.ones(256))  # 1D scalar
            
    model = DummyModel()
    
    try:
        factory = OptimizerFactory(model)
        groups = factory.split_param_groups()
        
        # Check structure
        required_keys = {'muon_matrix', 'adam_decay', 'adam_no_decay'}
        if set(groups.keys()) == required_keys:
            print("  ✓ Correct group keys")
        else:
            print(f"  ✗ Wrong keys: {groups.keys()}")
            all_passed = False
            
        # Count params
        total_params = sum(p.numel() for p in model.parameters())
        grouped_params = sum(sum(p.numel() for p in g) for g in groups.values())
        if total_params == grouped_params:
            print(f"  ✓ All {total_params} parameters accounted for")
        else:
            print(f"  ✗ Parameter count mismatch: {total_params} vs {grouped_params}")
            all_passed = False
            
    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False
    
    # Test 2: Newton-Schulz iteration
    print("\nTest 2: Newton-Schulz Orthogonalization")
    
    try:
        factory = OptimizerFactory(model)
        G = torch.randn(64, 32)
        G_orth = factory.newton_schulz_step(G, num_iters=5)
        
        # Check orthogonality: G_orth @ G_orth.T should be close to identity
        GGT = G_orth @ G_orth.T
        identity = torch.eye(G_orth.size(0))
        
        # Normalize for comparison (Newton-Schulz doesn't preserve scale)
        GGT_normalized = GGT / GGT.diag().mean()
        
        if torch.allclose(GGT_normalized, identity, atol=0.1):
            print("  ✓ Gradient approximately orthogonalized")
        else:
            print("  ✗ Orthogonalization not achieved")
            all_passed = False
            
    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False
    
    # Test 3: LR Schedule
    print("\nTest 3: Learning Rate Schedule")
    
    try:
        factory = OptimizerFactory(model)
        schedule = factory.get_lr_schedule(warmup_steps=100, total_steps=1000, init_lr_frac=0.1)
        
        # Check key points
        lr_0 = schedule(0)  # Should be init_lr_frac
        lr_warmup = schedule(100)  # Should be ~1.0
        lr_mid = schedule(550)  # Should be ~0.5
        lr_end = schedule(1000)  # Should be ~0.0
        
        checks = [
            (abs(lr_0 - 0.1) < 0.05, f"lr(0)={lr_0:.3f}, expected ~0.1"),
            (abs(lr_warmup - 1.0) < 0.05, f"lr(100)={lr_warmup:.3f}, expected ~1.0"),
            (abs(lr_mid - 0.5) < 0.1, f"lr(550)={lr_mid:.3f}, expected ~0.5"),
            (abs(lr_end) < 0.05, f"lr(1000)={lr_end:.3f}, expected ~0.0"),
        ]
        
        for passed, msg in checks:
            if passed:
                print(f"  ✓ {msg}")
            else:
                print(f"  ✗ {msg}")
                all_passed = False
                
    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! You've mastered the Optimization Layer!")
    else:
        print("❌ Some tests failed. Keep working on your implementation!")
    print("=" * 60)
    
    return all_passed


def peek():
    """Show the reference implementation from nanochat."""
    from nanochat import optim
    
    print("=" * 60)
    print("REFERENCE: Source code from nanochat/optim.py")
    print("=" * 60)
    
    # Show polar_express_coeffs
    print("\n### Polar Express Coefficients ###")
    print(f"polar_express_coeffs = {optim.polar_express_coeffs}")
    
    # Show muon_step_fused
    print("\n### muon_step_fused ###")
    print(inspect.getsource(optim.muon_step_fused))
    
    # Show MuonAdamW class
    print("\n### MuonAdamW.__init__ and step ###")
    print(inspect.getsource(optim.MuonAdamW.__init__))
    print(inspect.getsource(optim.MuonAdamW.step))


def main():
    """Run the puzzle interactively."""
    print(__doc__)
    print("=" * 60)
    print("Try implementing the OptimizerFactory class above!")
    print("Run with --verify to test, or --peek to see the solution.")
    print("=" * 60)
    
    # Quick smoke test
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 32)
    
    model = DummyModel()
    
    try:
        factory = OptimizerFactory(model)
        groups = factory.split_param_groups()
        print(f"\n✓ Param groups: {list(groups.keys())}")
        
        G = torch.randn(64, 32)
        G_orth = factory.newton_schulz_step(G)
        print(f"✓ Newton-Schulz output shape: {G_orth.shape}")
        
        schedule = factory.get_lr_schedule(100, 1000)
        print(f"✓ LR at step 0: {schedule(0):.4f}")
        print(f"✓ LR at step 500: {schedule(500):.4f}")
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
