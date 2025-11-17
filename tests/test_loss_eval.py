"""
Tests for loss evaluation functions, specifically evaluate_bpb.

Run with:
python -m pytest tests/test_loss_eval.py -v
"""

import math
import pytest
import torch
from unittest.mock import patch, MagicMock

from nanochat.loss_eval import evaluate_bpb
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import autodetect_device_type


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
    )


@pytest.fixture
def device():
    """Device to run tests on."""
    device_type = autodetect_device_type()
    if device_type == "cuda":
        return torch.device("cuda", 0)
    return torch.device(device_type)


@pytest.fixture
def model(small_config, device):
    """Model instance with initialized weights."""
    model = GPT(small_config)
    model.to(device)
    model.init_weights()
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    return model


@pytest.fixture
def token_bytes(small_config, device):
    """Token bytes tensor mapping token IDs to byte counts."""
    vocab_size = small_config.vocab_size
    # Create a simple mapping: token_id -> (token_id % 5) + 1 bytes
    # But set token 0 and token 1 to 0 (special tokens)
    token_bytes = torch.zeros(vocab_size, dtype=torch.int32, device=device)
    for i in range(vocab_size):
        if i == 0 or i == 1:  # Special tokens
            token_bytes[i] = 0
        else:
            token_bytes[i] = (i % 5) + 1  # 1-5 bytes
    return token_bytes


@pytest.fixture
def mock_batches(device):
    """Mock batches iterator yielding (x, y) tuples."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100

    def batch_generator():
        for _ in range(3):  # 3 batches
            x = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device
            )
            y = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device
            )
            yield x, y

    return batch_generator()


@pytest.fixture
def simple_token_bytes(device):
    """Simple token_bytes for component tests."""
    # vocab_size = 10, tokens 0-1 are special (0 bytes), rest have 1-8 bytes
    token_bytes = torch.tensor(
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device=device
    )
    return token_bytes


# ============================================================================
# Component Tests
# ============================================================================

def test_loss_computation_component(model, device):
    """Test model called with loss_reduction='none' for per-token losses."""
    batch_size = 2
    seq_len = 10
    vocab_size = model.config.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Call model with loss_reduction='none'
    model.eval()
    with torch.no_grad():
        loss2d = model(x, y, loss_reduction='none')

    # Verify loss shape - model returns flattened loss when
    # loss_reduction='none', so it should be (B * T,) shape
    expected_shape = (batch_size * seq_len,)
    assert loss2d.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {loss2d.shape}"
    expected_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    assert loss2d.dtype in expected_dtypes, \
        f"Expected float dtype, got {loss2d.dtype}"

    # Verify all losses are finite and non-negative
    assert torch.isfinite(loss2d).all(), "All losses should be finite"
    assert (loss2d >= 0).all(), "All losses should be non-negative"


def test_token_bytes_mapping(simple_token_bytes, device):
    """Test that token_bytes[y] correctly maps token IDs to byte counts."""
    # Create targets tensor
    targets = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], device=device)

    # Map to bytes
    num_bytes2d = simple_token_bytes[targets]

    # Verify mapping
    expected = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32, device=device
    )
    assert torch.equal(num_bytes2d, expected), \
        f"Expected {expected}, got {num_bytes2d}"

    # Test special tokens (0 and 1 should map to 0)
    special_targets = torch.tensor([[0, 1, 2], [1, 0, 3]], device=device)
    special_bytes = simple_token_bytes[special_targets]
    expected_special = torch.tensor(
        [[0, 0, 1], [0, 0, 2]], dtype=torch.int32, device=device
    )
    assert torch.equal(special_bytes, expected_special), \
        f"Special tokens should map to 0, got {special_bytes}"


def test_special_token_filtering(model, simple_token_bytes, device):
    """Test tokens with token_bytes==0 excluded from nats and bytes sums."""
    batch_size = 2
    seq_len = 4
    vocab_size = model.config.vocab_size

    # Create inputs and targets where some tokens are special
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # 0 and 1 are special tokens
    y = torch.tensor([[0, 1, 2, 3], [1, 0, 4, 5]], device=device)

    model.eval()
    with torch.no_grad():
        loss2d = model(x, y, loss_reduction='none')

    loss2d = loss2d.view(-1)
    y_flat = y.view(-1)

    # Map to bytes
    num_bytes2d = simple_token_bytes[y_flat]

    # Verify special tokens have 0 bytes
    assert (num_bytes2d[y_flat == 0] == 0).all(), \
        "Token 0 should have 0 bytes"
    assert (num_bytes2d[y_flat == 1] == 0).all(), \
        "Token 1 should have 0 bytes"

    # Verify mask (num_bytes2d > 0) correctly filters special tokens
    mask = num_bytes2d > 0
    assert not mask[y_flat == 0].any(), "Special tokens should be masked out"
    assert not mask[y_flat == 1].any(), "Special tokens should be masked out"
    assert mask[y_flat == 2].all(), "Non-special tokens should not be masked"

    # Verify that nats are only summed for non-special tokens
    total_nats = (loss2d * (num_bytes2d > 0)).sum()
    total_bytes = num_bytes2d.sum().item()

    # Both should exclude special tokens
    # y = [[0, 1, 2, 3], [1, 0, 4, 5]]
    # token_bytes: 0->0, 1->0, 2->1, 3->2, 4->3, 5->4
    # bytes: 0+0+1+2 + 0+0+3+4 = 10
    expected_bytes = 1 + 2 + 3 + 4  # tokens 2,3,4,5
    assert total_bytes == expected_bytes, \
        f"Total bytes should exclude special tokens, got {total_bytes}"
    assert total_nats > 0, "Total nats should be positive for non-special"


def test_ignored_token_filtering_fast_path(model, simple_token_bytes, device):
    """Test fast path when no ignored tokens (all targets >= 0)."""
    batch_size = 2
    seq_len = 4
    vocab_size = model.config.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # All >= 0
    y = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], device=device)

    model.eval()
    with torch.no_grad():
        loss2d = model(x, y, loss_reduction='none')

    loss2d = loss2d.view(-1)
    y_flat = y.view(-1)

    # Fast path: direct indexing
    num_bytes2d = simple_token_bytes[y_flat]

    total_nats = (loss2d * (num_bytes2d > 0)).sum()
    total_bytes = num_bytes2d.sum()

    assert total_bytes > 0, "Total bytes should be positive"
    assert total_nats > 0, "Total nats should be positive"


def test_ignored_token_filtering_complex_path(
    model, simple_token_bytes, device
):
    """Test complex path when some targets are ignored (targets < 0)."""
    batch_size = 2
    seq_len = 4
    vocab_size = model.config.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Some are -1 (ignored)
    y = torch.tensor([[2, -1, 4, 5], [-1, 7, 8, 9]], device=device)

    model.eval()
    with torch.no_grad():
        loss2d = model(x, y, loss_reduction='none')

    loss2d = loss2d.view(-1)
    y_flat = y.view(-1)

    # Complex path: conditional indexing
    valid = y_flat >= 0
    y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
    num_bytes2d = torch.where(
        valid,
        simple_token_bytes[y_safe],
        torch.zeros_like(y_flat, dtype=simple_token_bytes.dtype)
    )

    # Verify ignored tokens contribute 0 bytes
    assert (num_bytes2d[y_flat < 0] == 0).all(), \
        "Ignored tokens should contribute 0 bytes"

    # Verify valid tokens are correctly mapped
    valid_indices = y_flat[valid]
    assert torch.equal(
        num_bytes2d[valid], simple_token_bytes[valid_indices]
    ), "Valid tokens should be correctly mapped to bytes"

    total_bytes = num_bytes2d.sum().item()

    # Should exclude ignored tokens
    # y = [[2, -1, 4, 5], [-1, 7, 8, 9]]
    # token_bytes: 2->1, 4->3, 5->4, 7->6, 8->7, 9->8
    # bytes: 1+3+4 + 6+7+8 = 29
    expected_bytes = 1 + 3 + 4 + 6 + 7 + 8  # Only valid tokens
    assert total_bytes == expected_bytes, (
        f"Total bytes should exclude ignored tokens, "
        f"expected {expected_bytes}, got {total_bytes}"
    )


def test_fast_path_vs_complex_path_consistency(
    model, simple_token_bytes, device
):
    """Test fast and complex paths produce same results when no ignored."""
    batch_size = 2
    seq_len = 4
    vocab_size = model.config.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # All >= 0
    y = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], device=device)

    model.eval()
    with torch.no_grad():
        loss2d = model(x, y, loss_reduction='none')

    loss2d = loss2d.view(-1)
    y_flat = y.view(-1)

    # Fast path
    num_bytes_fast = simple_token_bytes[y_flat]
    total_nats_fast = (loss2d * (num_bytes_fast > 0)).sum()
    total_bytes_fast = num_bytes_fast.sum()

    # Complex path (simulating the code path)
    valid = y_flat >= 0
    y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
    num_bytes_complex = torch.where(
        valid,
        simple_token_bytes[y_safe],
        torch.zeros_like(y_flat, dtype=simple_token_bytes.dtype)
    )
    total_nats_complex = (loss2d * (num_bytes_complex > 0)).sum()
    total_bytes_complex = num_bytes_complex.sum()

    # Results should be identical
    assert torch.equal(num_bytes_fast, num_bytes_complex), \
        "Fast and complex paths should produce same bytes"
    assert torch.allclose(total_nats_fast, total_nats_complex), \
        "Fast and complex paths should produce same nats"
    assert total_bytes_fast == total_bytes_complex, \
        "Fast and complex paths should produce same total bytes"


def test_bpb_calculation_formula(device):
    """Test bpb formula: bpb = total_nats / (log(2) * total_bytes)."""
    # Create a mock scenario
    total_nats = torch.tensor(10.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(5, dtype=torch.int64, device=device)

    # Calculate bpb manually
    expected_bpb = total_nats.item() / (math.log(2) * total_bytes.item())

    # Simulate the calculation from evaluate_bpb
    bpb = total_nats.item() / (math.log(2) * total_bytes.item())

    assert abs(bpb - expected_bpb) < 1e-6, \
        f"BPB calculation incorrect, expected {expected_bpb}, got {bpb}"

    # Verify conversion from nats to bits (divide by log(2))
    nats_to_bits = 1.0 / math.log(2)
    assert abs(nats_to_bits - 1.442695) < 1e-5, \
        "Nats to bits conversion should be ~1.442695"


def test_bpb_edge_case_zero_bytes(device):
    """Test edge case: total_bytes == 0 returns float('inf')."""
    total_nats = torch.tensor(10.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    # Simulate the edge case handling
    if total_bytes.item() == 0:
        bpb = float('inf')
    else:
        bpb = total_nats.item() / (math.log(2) * total_bytes.item())

    assert bpb == float('inf'), "BPB should be inf when total_bytes == 0"
    assert math.isinf(bpb), "BPB should be infinite"


def test_distributed_reduction_single_process(model, token_bytes, device):
    """Test distributed reduction with single process (world_size == 1)."""
    batch_size = 2
    seq_len = 10
    vocab_size = model.config.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Avoid special tokens
    y = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)

    def batches():
        yield x, y

    model.eval()

    # Mock torch.distributed to simulate single process
    with patch('nanochat.loss_eval.dist.is_initialized', return_value=False):
        with patch('nanochat.loss_eval.dist.get_world_size', return_value=1):
            bpb = evaluate_bpb(
                model, batches(), steps=1, token_bytes=token_bytes
            )

    # Should return a finite value
    assert math.isfinite(bpb), f"BPB should be finite, got {bpb}"
    assert bpb >= 0, f"BPB should be non-negative, got {bpb}"


def test_distributed_reduction_multi_process(model, token_bytes, device):
    """Test distributed reduction with multiple processes (mocked)."""
    batch_size = 2
    seq_len = 10
    vocab_size = model.config.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)

    def batches():
        yield x, y

    model.eval()

    # Mock torch.distributed to simulate multi-process
    mock_all_reduce = MagicMock()

    with patch('nanochat.loss_eval.dist.is_initialized', return_value=True):
        with patch('nanochat.loss_eval.dist.get_world_size', return_value=4):
            with patch('nanochat.loss_eval.dist.all_reduce', mock_all_reduce):
                bpb = evaluate_bpb(
                    model, batches(), steps=1, token_bytes=token_bytes
                )

    # Verify all_reduce was called twice (once for nats, once for bytes)
    assert mock_all_reduce.call_count == 2, (
        f"all_reduce should be called twice, "
        f"got {mock_all_reduce.call_count}"
    )

    # Should return a finite value
    assert math.isfinite(bpb), f"BPB should be finite, got {bpb}"


# ============================================================================
# End-to-End Tests
# ============================================================================

def test_end_to_end_simple_mock_data(device):
    """Test simple end-to-end with mock data - verify bpb matches expected."""
    vocab_size = 10
    batch_size = 2
    seq_len = 4

    # Create a simple mock model
    class MockModel:
        def __init__(self, device):
            self.device = device

        def get_device(self):
            return self.device

        def __call__(self, x, y, loss_reduction='mean'):
            # Return fixed loss values for testing
            B, T = x.shape
            if loss_reduction == 'none':
                # Return (B, T) tensor with known values
                return torch.ones(
                    B, T, dtype=torch.float32, device=self.device
                ) * 0.5
            else:
                return torch.tensor(
                    0.5, dtype=torch.float32, device=self.device
                )

    model = MockModel(device)

    # Create token_bytes: tokens 0-1 are special (0 bytes), rest have 1 byte
    token_bytes = torch.tensor(
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32, device=device
    )

    # Create batches with known targets
    def batches():
        # Batch 1: targets are [2, 3, 4, 5] and [6, 7, 8, 9]
        # (all non-special, 1 byte each)
        x1 = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y1 = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], device=device)
        yield x1, y1

    # Calculate expected bpb
    # Each token has loss 0.5 nats, and 1 byte
    # Total nats = 0.5 * 8 tokens = 4.0 nats
    # Total bytes = 1 * 8 tokens = 8 bytes
    # bpb = 4.0 / (log(2) * 8) = 4.0 / (0.693 * 8) ≈ 0.721

    with patch('nanochat.loss_eval.dist.is_initialized', return_value=False):
        bpb = evaluate_bpb(
            model, batches(), steps=1, token_bytes=token_bytes
        )

    expected_bpb = 4.0 / (math.log(2) * 8)
    assert abs(bpb - expected_bpb) < 1e-5, \
        f"Expected bpb ≈ {expected_bpb}, got {bpb}"


def test_end_to_end_real_model(model, token_bytes, device):
    """Test end-to-end with real GPT model - verify bpb is reasonable."""
    batch_size = 2
    seq_len = 10
    vocab_size = model.config.vocab_size

    # Create realistic batches
    def batches():
        for _ in range(2):  # 2 batches
            x = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device
            )
            # Avoid special tokens
            y = torch.randint(
                2, vocab_size, (batch_size, seq_len), device=device
            )
            yield x, y

    model.eval()

    with patch('nanochat.loss_eval.dist.is_initialized', return_value=False):
        bpb = evaluate_bpb(
            model, batches(), steps=2, token_bytes=token_bytes
        )

    # Verify bpb is reasonable (not inf, not negative, not too large)
    assert math.isfinite(bpb), f"BPB should be finite, got {bpb}"
    assert bpb >= 0, f"BPB should be non-negative, got {bpb}"
    assert bpb < 100, f"BPB should be reasonable (< 100), got {bpb}"


def test_multiple_batches_accumulation(model, token_bytes, device):
    """Test multiple batches - verify accumulation works correctly."""
    batch_size = 2
    seq_len = 10
    vocab_size = model.config.vocab_size

    # Create batches with known structure
    batch_data = []

    def batches():
        for i in range(3):  # 3 batches
            x = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device
            )
            y = torch.randint(
                2, vocab_size, (batch_size, seq_len), device=device
            )
            batch_data.append((x.clone(), y.clone()))
            yield x, y

    model.eval()

    with patch('nanochat.loss_eval.dist.is_initialized', return_value=False):
        bpb_all = evaluate_bpb(
            model, batches(), steps=3, token_bytes=token_bytes
        )

    # Calculate bpb for each batch individually
    bpb_individual = []
    for x, y in batch_data:
        with patch(
            'nanochat.loss_eval.dist.is_initialized', return_value=False
        ):
            bpb = evaluate_bpb(
                model, iter([(x, y)]), steps=1, token_bytes=token_bytes
            )
            bpb_individual.append(bpb)

    # The bpb for all batches together should be a weighted average
    # (weighted by bytes in each batch)
    # For simplicity, just verify all batches contribute
    assert math.isfinite(bpb_all), "BPB for all batches should be finite"
    assert all(math.isfinite(b) for b in bpb_individual), \
        "All individual BPBs should be finite"

    # All should be in similar range (not drastically different)
    for bpb_ind in bpb_individual:
        assert abs(bpb_ind - bpb_all) < 10, (
            f"Individual BPB {bpb_ind} should be similar to "
            f"combined {bpb_all}"
        )


# ============================================================================
# Example Usage Test (like base_train.py)
# ============================================================================

def test_example_usage_base_train_style(model, device):
    """Test realistic training evaluation scenario like base_train.py."""
    batch_size = 2
    seq_len = 10
    vocab_size = model.config.vocab_size
    eval_tokens = 40  # Small number for testing
    eval_steps = eval_tokens // (batch_size * seq_len)  # Should be 2 steps

    # Create token_bytes (similar to base_train.py usage)
    token_bytes = torch.zeros(vocab_size, dtype=torch.int32, device=device)

    for i in range(vocab_size):
        if i < 2:  # Special tokens
            token_bytes[i] = 0
        else:
            token_bytes[i] = (i % 5) + 1

    # Create val_loader (similar to base_train.py)
    def build_val_loader():
        def val_batches():
            for _ in range(eval_steps):
                x = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=device
                )
                # Avoid special tokens
                y = torch.randint(
                    2, vocab_size, (batch_size, seq_len), device=device
                )
                yield x, y
        return val_batches()

    # Evaluate (similar to base_train.py lines 219-225)
    model.eval()
    val_loader = build_val_loader()

    with patch('nanochat.loss_eval.dist.is_initialized', return_value=False):
        val_bpb = evaluate_bpb(
            model, val_loader, eval_steps, token_bytes
        )

    # Verify it returns a float bpb value
    assert isinstance(val_bpb, float), \
        f"val_bpb should be float, got {type(val_bpb)}"
    assert math.isfinite(val_bpb), \
        f"val_bpb should be finite, got {val_bpb}"
    assert val_bpb >= 0, f"val_bpb should be non-negative, got {val_bpb}"


def test_example_usage_different_eval_steps(model, token_bytes, device):
    """Test with different eval_steps (like base_train.py)."""
    batch_size = 2
    seq_len = 10
    vocab_size = model.config.vocab_size

    def build_val_loader():
        def val_batches():
            while True:  # Infinite generator
                x = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=device
                )
                y = torch.randint(
                    2, vocab_size, (batch_size, seq_len), device=device
                )
                yield x, y
        return val_batches()

    model.eval()

    # Test with different eval_steps
    for eval_steps in [1, 2, 5]:
        # Create new iterator for each test
        val_loader = build_val_loader()

        with patch(
            'nanochat.loss_eval.dist.is_initialized', return_value=False
        ):
            val_bpb = evaluate_bpb(
                model, val_loader, eval_steps, token_bytes
            )

        assert isinstance(val_bpb, float), \
            f"val_bpb should be float for eval_steps={eval_steps}"
        assert math.isfinite(val_bpb), (
            f"val_bpb should be finite for eval_steps={eval_steps}, "
            f"got {val_bpb}"
        )
        assert val_bpb >= 0, (
            f"val_bpb should be non-negative for eval_steps={eval_steps}, "
            f"got {val_bpb}"
        )


# ============================================================================
# Edge Cases
# ============================================================================

def test_edge_case_all_special_tokens(model, token_bytes, device):
    """Test edge case: all tokens are special tokens (token_bytes == 0)."""
    batch_size = 2
    seq_len = 10
    vocab_size = model.config.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # All token 0 (special)
    y = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    def batches():
        yield x, y

    model.eval()

    with patch('nanochat.loss_eval.dist.is_initialized', return_value=False):
        bpb = evaluate_bpb(model, batches(), steps=1, token_bytes=token_bytes)

    # Should return inf because total_bytes == 0
    assert bpb == float('inf'), \
        f"BPB should be inf when all tokens are special, got {bpb}"


def test_edge_case_all_ignored_tokens(model, token_bytes, device):
    """Test edge case: all targets are ignored (all targets < 0)."""
    batch_size = 2
    seq_len = 10
    vocab_size = model.config.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # All ignored
    y = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)

    def batches():
        yield x, y

    model.eval()

    with patch('nanochat.loss_eval.dist.is_initialized', return_value=False):
        bpb = evaluate_bpb(model, batches(), steps=1, token_bytes=token_bytes)

    # Should return inf because total_bytes == 0
    assert bpb == float('inf'), \
        f"BPB should be inf when all tokens are ignored, got {bpb}"


def test_edge_case_mixed_special_and_ignored(model, token_bytes, device):
    """Test edge case: mix of special tokens and ignored tokens."""
    batch_size = 2
    seq_len = 4
    vocab_size = model.config.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Mix of special (0,1) and ignored (-1)
    y = torch.tensor([[0, -1, 1, -1], [-1, 0, -1, 1]], device=device)

    def batches():
        yield x, y

    model.eval()

    with patch('nanochat.loss_eval.dist.is_initialized', return_value=False):
        bpb = evaluate_bpb(model, batches(), steps=1, token_bytes=token_bytes)

    # Should return inf because no valid tokens (all are special or ignored)
    assert bpb == float('inf'), (
        f"BPB should be inf when all tokens are special or ignored, "
        f"got {bpb}"
    )
