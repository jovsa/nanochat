"""
Puzzle 6: The Evaluator ⭐⭐⭐⭐
Goal: Build the evaluation system with CORE metric.

Usage (from repo root):
python -m puzzles.06_evaluator          # Run the puzzle
python -m puzzles.06_evaluator --verify # Verify your solution
python -m puzzles.06_evaluator --peek   # See reference solution (oracle)
"""

import argparse
import torch
import inspect
from typing import Optional
from jinja2 import Template

# =============================================================================
# YOUR TASK: Implement the CORE_Evaluator class
# =============================================================================

class CORE_Evaluator:
    """
    Build the evaluation system using CORE metric (from DCLM paper).
    
    Your implementation should:
    1. Batching: Group MC options by common prefix/suffix
    2. Rendering: Jinja2 logic for MC/Schema/LM tasks
    3. Scoring: Loss-based selection logic
    4. Aggregation: DDP reduction of results
    
    Hints:
    - Study nanochat/core_eval.py
    - MC = Multiple Choice (pick best option by lowest loss)
    - Schema = Fill in the blank with predefined answers
    - LM = Language modeling (perplexity-based)
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def render_prompts_mc(self, item: dict, 
                          continuation_delimiter: str = " ",
                          fewshot_examples: Optional[list] = None) -> list[str]:
        """
        Render prompts for a multiple choice question.
        
        Args:
            item: Dict with 'query', 'choices' (list of options), 'gold' (correct idx)
            continuation_delimiter: String between question and answer
            fewshot_examples: Optional few-shot examples
            
        Returns:
            List of complete prompts (one per choice)
            
        Example:
            item = {'query': 'Capital of France?', 'choices': ['Paris', 'London'], 'gold': 0}
            -> ['Capital of France? Paris', 'Capital of France? London']
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement MC prompt rendering")
    
    def find_common_length(self, token_sequences: list[list[int]], 
                           direction: str = 'left') -> int:
        """
        Find the length of common prefix or suffix across sequences.
        
        Args:
            token_sequences: List of tokenized sequences
            direction: 'left' for prefix, 'right' for suffix
            
        Returns:
            Length of common prefix/suffix
            
        This is used to efficiently batch MC options that share context.
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement common length finding")
    
    def forward_model(self, input_ids: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning per-token losses and predictions.
        
        Args:
            input_ids: Tensor of shape (B, T)
            
        Returns:
            tuple: (losses, predictions)
                - losses: shape (B, T), cross-entropy loss per token
                - predictions: shape (B, T), argmax predictions
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement model forward")
    
    def evaluate_example_mc(self, item: dict) -> bool:
        """
        Evaluate a single MC example.
        
        Args:
            item: MC item with query, choices, gold
            
        Returns:
            True if model picks correct answer, False otherwise
            
        Scoring:
        - Compute loss for each choice continuation
        - Pick choice with lowest average loss on continuation tokens
        - Compare to gold answer
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement MC evaluation")
    
    def evaluate_task(self, data: list[dict], task_type: str = "mc") -> float:
        """
        Evaluate a full task and return accuracy.
        
        Args:
            data: List of evaluation items
            task_type: "mc", "schema", or "lm"
            
        Returns:
            Accuracy (0.0 to 1.0)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement task evaluation")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify():
    print("=" * 60)
    print("VERIFICATION: Testing your CORE_Evaluator")
    print("=" * 60)
    
    all_passed = True
    
    print("\nTest 1: Common Length Finding")
    try:
        class MockObjects:
            pass
        evaluator = CORE_Evaluator(MockObjects(), MockObjects())
        
        seqs = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [1, 2, 3, 8, 9]]
        common_left = evaluator.find_common_length(seqs, 'left')
        if common_left == 3:
            print("  ✓ Common prefix length correct: 3")
        else:
            print(f"  ✗ Wrong: {common_left}, expected 3")
            all_passed = False
            
        seqs = [[1, 2, 8, 9, 10], [3, 4, 8, 9, 10], [5, 6, 8, 9, 10]]
        common_right = evaluator.find_common_length(seqs, 'right')
        if common_right == 3:
            print("  ✓ Common suffix length correct: 3")
        else:
            print(f"  ✗ Wrong: {common_right}, expected 3")
            all_passed = False
            
    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False
    
    print("\nTest 2: MC Prompt Rendering")
    try:
        item = {'query': 'Q: What is 2+2?', 'choices': ['4', '5', '22'], 'gold': 0}
        prompts = evaluator.render_prompts_mc(item)
        
        if len(prompts) == 3:
            print("  ✓ Correct number of prompts")
        else:
            print(f"  ✗ Wrong count: {len(prompts)}")
            all_passed = False
            
        if 'Q: What is 2+2?' in prompts[0] and '4' in prompts[0]:
            print("  ✓ First prompt contains query and choice")
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
    from nanochat import core_eval
    print("=" * 60)
    print("REFERENCE: nanochat/core_eval.py")
    print("=" * 60)
    print("\n### render_prompts_mc ###")
    print(inspect.getsource(core_eval.render_prompts_mc))
    print("\n### find_common_length ###")
    print(inspect.getsource(core_eval.find_common_length))
    print("\n### evaluate_example ###")
    print(inspect.getsource(core_eval.evaluate_example))


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
