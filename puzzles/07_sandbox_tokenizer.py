"""
Puzzle 7: The Sandbox & Tokenizer ⭐⭐⭐⭐⭐
Goal: Build the safety and text processing layer.

Usage:
python puzzles/07_sandbox_tokenizer.py          # Run the puzzle
python puzzles/07_sandbox_tokenizer.py --verify # Verify your solution
python puzzles/07_sandbox_tokenizer.py --peek   # See reference solution (oracle)
"""

import argparse
import inspect
import multiprocessing
import os
from typing import Optional
from dataclasses import dataclass

# =============================================================================
# YOUR TASK: Implement the Tokenizer and Sandbox classes
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of executing Python code in a sandbox."""
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False


class TokenizerWrapper:
    """
    Wrapper around tokenizer with tool-use support.
    
    Your implementation should:
    1. Handle special tokens like <|python_start|>, <|python_end|>
    2. Render conversations for training and completion
    3. Support prepending BOS token during encoding
    
    Hints:
    - Study nanochat/tokenizer.py: RustBPETokenizer
    - Special tokens are encoded differently than regular text
    """
    
    def __init__(self, base_tokenizer):
        self.enc = base_tokenizer
        
    def encode(self, text: str, prepend: Optional[int] = None) -> list[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: String to encode
            prepend: Optional token ID to prepend (e.g., BOS)
            
        Returns:
            List of token IDs
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement encoding")
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to string."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement decoding")
    
    def encode_special(self, token_name: str) -> int:
        """
        Encode a special token like <|python_start|>.
        
        Args:
            token_name: The special token string
            
        Returns:
            Single token ID
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement special token encoding")
    
    def render_for_training(self, conversation: list[dict]
                            ) -> tuple[list[int], list[float]]:
        """
        Render a conversation for SFT training.
        
        Args:
            conversation: List of {'role': 'user'/'assistant', 'content': str}
            
        Returns:
            tuple: (tokens, mask)
                - tokens: All token IDs
                - mask: 1.0 for assistant tokens (train), 0.0 for user (no train)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement training rendering")
    
    def render_for_completion(self, conversation: list[dict]) -> list[int]:
        """
        Render a conversation for inference (priming assistant response).
        
        Args:
            conversation: Conversation history
            
        Returns:
            Token IDs ready for generation
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement completion rendering")


class Sandbox:
    """
    Sandboxed Python code execution with safety guards.
    
    Your implementation should:
    1. Use multiprocessing for isolation
    2. Apply resource limits (time, memory)
    3. Block dangerous operations (file access, network, etc.)
    
    Hints:
    - Study nanochat/execution.py: execute_code, reliability_guard
    - Use subprocess or multiprocessing for isolation
    - Set resource limits using resource module on Linux
    """
    
    def __init__(self, timeout: float = 5.0, 
                 max_memory_bytes: int = 256 * 1024 * 1024):
        self.timeout = timeout
        self.max_memory_bytes = max_memory_bytes
    
    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code in a sandboxed environment.
        
        Args:
            code: Python code string
            
        Returns:
            ExecutionResult with success status and outputs
            
        Safety measures:
        - Timeout to prevent infinite loops
        - Memory limit to prevent OOM
        - Block dangerous builtins (exec, eval, __import__, open, etc.)
        - Block system calls (os.system, subprocess, etc.)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement sandboxed execution")
    
    def _apply_security_guards(self):
        """
        Apply security restrictions in the subprocess.
        Should be called at the start of the sandboxed process.
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement security guards")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify():
    print("=" * 60)
    print("VERIFICATION: Testing Sandbox & Tokenizer")
    print("=" * 60)
    
    all_passed = True
    
    print("\nTest 1: Sandbox - Safe Execution")
    try:
        sandbox = Sandbox(timeout=2.0)
        result = sandbox.execute("print('hello world')")
        
        if result.success and 'hello world' in result.stdout:
            print("  ✓ Basic execution works")
        else:
            print(f"  ✗ Failed: {result}")
            all_passed = False
            
        result = sandbox.execute("x = 1 + 1\nprint(x)")
        if result.success and '2' in result.stdout:
            print("  ✓ Arithmetic works")
        else:
            all_passed = False
            
    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False
    
    print("\nTest 2: Sandbox - Security")
    try:
        sandbox = Sandbox(timeout=1.0)
        
        # These should all fail or be blocked
        dangerous = [
            "import os; os.system('echo pwned')",
            "open('/etc/passwd').read()",
            "__import__('subprocess').run(['ls'])",
        ]
        
        blocked = 0
        for code in dangerous:
            result = sandbox.execute(code)
            if not result.success:
                blocked += 1
                
        if blocked == len(dangerous):
            print(f"  ✓ All {blocked} dangerous operations blocked")
        else:
            print(f"  ✗ Only {blocked}/{len(dangerous)} blocked")
            all_passed = False
            
    except NotImplementedError as e:
        print(f"  ✗ Not implemented: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_passed = False
    
    print("\nTest 3: Sandbox - Timeout")
    try:
        sandbox = Sandbox(timeout=0.5)
        result = sandbox.execute("while True: pass")
        
        if result.timeout:
            print("  ✓ Infinite loop timed out correctly")
        else:
            print("  ✗ Timeout not triggered")
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
    from nanochat import execution, tokenizer
    print("=" * 60)
    print("REFERENCE: nanochat/execution.py & tokenizer.py")
    print("=" * 60)
    print("\n### execute_code ###")
    print(inspect.getsource(execution.execute_code))
    print("\n### reliability_guard ###")
    print(inspect.getsource(execution.reliability_guard))


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
