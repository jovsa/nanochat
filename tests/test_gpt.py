"""
Tests for GPT model.

Run with:
python -m pytest tests/test_gpt.py -v

To compare with an alternative model implementation, set ALTERNATIVE_MODEL_CLASS
at the top of this file (see the import section).
"""

import pytest
import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig, CausalSelfAttention, MLP, Block, apply_rotary_emb, norm
from nanochat.common import autodetect_device_type
from nanochat.engine import KVCache

# ============================================================================
# Alternative Model Import (for comparison tests)
# ============================================================================
# To compare with an alternative model implementation, uncomment and modify:
# ALTERNATIVE_MODEL_CLASS = jgpt
from nanochat.jgpt import JGPT # or wherever your alternative model is
ALTERNATIVE_MODEL_CLASS = JGPT # Set this to your alternative model class


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
def gqa_config():
    """Config with Group Query Attention (GQA) - different n_head and n_kv_head."""
    return GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
    )


@pytest.fixture
def device():
    """Device to run tests on."""
    device_type = autodetect_device_type()
    if device_type == "cuda":
        # Return the actual CUDA device (cuda:0) not just "cuda"
        return torch.device("cuda", 0)
    return torch.device(device_type)


@pytest.fixture
def model(small_config, device):
    """Model instance with initialized weights."""
    model = GPT(small_config)
    model.to(device)
    model.init_weights()
    # On CUDA, init_weights() casts embedding to bfloat16, but linear layers stay float32
    # This causes dtype mismatches. For tests, ensure all parameters are the same dtype.
    if device.type == "cuda":
        # Cast all parameters to bfloat16 to match the embedding
        model = model.to(dtype=torch.bfloat16)
    return model


@pytest.fixture
def sample_input(small_config, device):
    """Sample input tensor."""
    batch_size = 2
    seq_len = 10
    return torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device=device)


@pytest.fixture
def sample_targets(sample_input):
    """Sample targets tensor (shifted input, same length as input)."""
    # Targets should be the same length as input for cross-entropy
    # Standard pattern: predict next token at each position
    # For input [a, b, c, d], we predict [b, c, d, next] where next is the actual next token
    # Since we don't have the actual next token, we'll use shifted input and pad/ignore the last position
    targets = torch.roll(sample_input, shifts=-1, dims=1)
    # The last position doesn't have a target, so we can set it to -1 (ignore_index)
    targets[:, -1] = -1
    return targets


@pytest.fixture
def kv_cache(small_config, device):
    """KV cache instance."""
    batch_size = 1
    num_heads = small_config.n_kv_head
    seq_len = small_config.sequence_len
    head_dim = small_config.n_embd // small_config.n_head
    num_layers = small_config.n_layer
    return KVCache(batch_size, num_heads, seq_len, head_dim, num_layers)


@pytest.fixture
def alternative_model_class():
    """
    Optional fixture for alternative model class to compare against.
    Set ALTERNATIVE_MODEL_CLASS at module level (imported above) to use.
    """
    return ALTERNATIVE_MODEL_CLASS


@pytest.fixture
def alternative_model(alternative_model_class, small_config, device):
    """
    Optional fixture for alternative model instance to compare against.
    Only created if alternative_model_class is provided.
    """
    if alternative_model_class is None:
        return None

    try:
        model = alternative_model_class(small_config)
        model.to(device)
        # Try to initialize weights if the method exists
        if hasattr(model, 'init_weights'):
            model.init_weights()
        # Copy weights from reference model to ensure fair comparison
        # This will be done in the comparison tests
        return model
    except Exception as e:
        pytest.skip(f"Could not create alternative model instance: {e}")


# ============================================================================
# Model Initialization Tests
# ============================================================================

def test_gpt_init(small_config, device):
    """Test GPT initialization creates model with correct structure."""
    model = GPT(small_config)
    model.to(device)

    # Check structure
    assert hasattr(model, 'transformer')
    assert hasattr(model, 'lm_head')
    assert hasattr(model, 'cos')
    assert hasattr(model, 'sin')

    # Check transformer components
    assert 'wte' in model.transformer
    assert 'h' in model.transformer
    assert len(model.transformer.h) == small_config.n_layer

    # Check embedding
    assert model.transformer.wte.num_embeddings == small_config.vocab_size
    assert model.transformer.wte.embedding_dim == small_config.n_embd

    # Check lm_head
    assert model.lm_head.in_features == small_config.n_embd
    assert model.lm_head.out_features == small_config.vocab_size

    # Check rotary embeddings are registered
    assert 'cos' in dict(model.named_buffers())
    assert 'sin' in dict(model.named_buffers())


def test_init_weights(small_config, device):
    """Test init_weights initializes weights correctly."""
    model = GPT(small_config)
    model.to(device)
    model.init_weights()

    # Check lm_head is zeroed
    assert torch.allclose(model.lm_head.weight, torch.zeros_like(model.lm_head.weight))

    # Check c_proj weights are zeroed in all blocks
    for block in model.transformer.h:
        assert torch.allclose(block.attn.c_proj.weight, torch.zeros_like(block.attn.c_proj.weight))
        assert torch.allclose(block.mlp.c_proj.weight, torch.zeros_like(block.mlp.c_proj.weight))

    # Check rotary embeddings are in bfloat16
    assert model.cos.dtype == torch.bfloat16
    assert model.sin.dtype == torch.bfloat16

    # Check rotary embeddings shape
    head_dim = small_config.n_embd // small_config.n_head
    expected_seq_len = small_config.sequence_len * 10
    assert model.cos.shape == (1, expected_seq_len, 1, head_dim // 2)
    assert model.sin.shape == (1, expected_seq_len, 1, head_dim // 2)


def test_get_device(model, device):
    """Test get_device returns correct device."""
    # Compare device types and indices, not exact device objects
    model_device = model.get_device()
    assert model_device.type == device.type
    if device.type == "cuda":
        # For CUDA, both should have an index
        assert hasattr(model_device, 'index')
        assert hasattr(device, 'index')
        # They should match (both should be cuda:0)
        assert model_device.index == device.index


# ============================================================================
# Forward Pass Tests
# ============================================================================

def test_forward_training_mode(model, sample_input, sample_targets):
    """Test forward pass in training mode returns loss."""
    model.train()
    loss = model.forward(sample_input, targets=sample_targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0  # loss should be non-negative


def test_forward_inference_mode(model, sample_input):
    """Test forward pass in inference mode returns logits."""
    model.eval()
    logits = model.forward(sample_input)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (sample_input.shape[0], sample_input.shape[1], model.config.vocab_size)
    # Check logits are softcapped (should be in range [-15, 15])
    assert torch.all(logits >= -15.0)
    assert torch.all(logits <= 15.0)


def test_forward_different_batch_sizes(model, small_config, device):
    """Test forward pass with different batch sizes."""
    seq_len = 10
    vocab_size = small_config.vocab_size

    for batch_size in [1, 2, 4]:
        idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits = model.forward(idx)
        assert logits.shape == (batch_size, seq_len, vocab_size)


def test_forward_different_sequence_lengths(model, small_config, device):
    """Test forward pass with different sequence lengths."""
    batch_size = 2
    vocab_size = small_config.vocab_size

    for seq_len in [1, 5, 10, 20]:
        idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits = model.forward(idx)
        assert logits.shape == (batch_size, seq_len, vocab_size)


def test_forward_loss_reduction_mean(model, sample_input, sample_targets):
    """Test loss_reduction='mean'."""
    model.train()
    loss_mean = model.forward(sample_input, targets=sample_targets, loss_reduction='mean')
    assert loss_mean.dim() == 0  # scalar


def test_forward_loss_reduction_sum(model, sample_input, sample_targets):
    """Test loss_reduction='sum'."""
    model.train()
    loss_sum = model.forward(sample_input, targets=sample_targets, loss_reduction='sum')
    assert loss_sum.dim() == 0  # scalar
    # Sum should be larger than mean (for non-trivial case)
    loss_mean = model.forward(sample_input, targets=sample_targets, loss_reduction='mean')
    # They should be different unless loss is zero
    if loss_mean.item() > 0:
        assert loss_sum.item() > loss_mean.item()


def test_forward_ignore_index(model, sample_input, device):
    """Test ignore_index handling in loss computation."""
    model.train()
    # Create targets with same length as input, with some -1 values (ignore_index)
    targets = torch.roll(sample_input, shifts=-1, dims=1)  # Shifted by 1
    targets[0, 0] = -1  # Mark first token as ignored
    targets[1, 2] = -1  # Mark another token as ignored
    targets[:, -1] = -1  # Last position doesn't have a target

    loss = model.forward(sample_input, targets=targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_forward_logits_softcap(model, sample_input):
    """Test logits softcap is applied."""
    model.eval()
    logits = model.forward(sample_input)

    # Logits should be softcapped to [-15, 15]
    assert torch.all(logits >= -15.0)
    assert torch.all(logits <= 15.0)

    # Check that tanh softcap is applied (values should be in range)
    # If we had unbounded logits, some would be outside this range
    # This is a sanity check that softcap is working


# ============================================================================
# Generation Tests
# ============================================================================

def test_generate_produces_tokens(model, device):
    """Test generate method produces tokens."""
    tokens = [1, 2, 3]
    max_tokens = 5

    generated = list(model.generate(tokens, max_tokens=max_tokens))

    assert len(generated) == max_tokens
    assert all(isinstance(t, int) for t in generated)
    assert all(0 <= t < model.config.vocab_size for t in generated)


def test_generate_respects_max_tokens(model, device):
    """Test generation respects max_tokens limit."""
    tokens = [1, 2, 3]

    for max_tokens in [1, 3, 5, 10]:
        generated = list(model.generate(tokens, max_tokens=max_tokens))
        assert len(generated) == max_tokens


def test_generate_temperature_zero(model, device):
    """Test temperature=0 (greedy decoding)."""
    tokens = [1, 2, 3]
    max_tokens = 3

    # With temperature=0, should always pick argmax (deterministic)
    generated1 = list(model.generate(tokens, max_tokens=max_tokens, temperature=0.0))
    generated2 = list(model.generate(tokens, max_tokens=max_tokens, temperature=0.0))

    assert generated1 == generated2  # Should be deterministic


def test_generate_temperature_positive(model, device):
    """Test temperature>0 (sampling)."""
    tokens = [1, 2, 3]
    max_tokens = 3

    # With temperature>0, should sample (may be non-deterministic)
    generated = list(model.generate(tokens, max_tokens=max_tokens, temperature=1.0))
    assert len(generated) == max_tokens
    assert all(0 <= t < model.config.vocab_size for t in generated)


def test_generate_top_k(model, device):
    """Test top_k filtering."""
    tokens = [1, 2, 3]
    max_tokens = 3
    top_k = 10

    generated = list(model.generate(tokens, max_tokens=max_tokens, top_k=top_k))
    assert len(generated) == max_tokens


def test_generate_seed_reproducibility(model, device):
    """Test seed reproducibility."""
    tokens = [1, 2, 3]
    max_tokens = 5
    seed = 42

    generated1 = list(model.generate(tokens, max_tokens=max_tokens, temperature=1.0, seed=seed))
    generated2 = list(model.generate(tokens, max_tokens=max_tokens, temperature=1.0, seed=seed))

    assert generated1 == generated2  # Should be reproducible with same seed


# ============================================================================
# KV Cache Integration Tests
# ============================================================================

def test_forward_with_kv_cache(model, kv_cache, device):
    """Test forward pass with KV cache (inference scenario)."""
    model.eval()
    batch_size = 1
    seq_len = 5
    idx = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    logits = model.forward(idx, kv_cache=kv_cache)

    assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
    assert kv_cache.get_pos() == seq_len  # Position should advance


def test_kv_cache_position_tracking(model, kv_cache, device):
    """Test KV cache position tracking."""
    model.eval()
    batch_size = 1
    seq_len = 5
    idx = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    initial_pos = kv_cache.get_pos()
    assert initial_pos == 0

    model.forward(idx, kv_cache=kv_cache)
    assert kv_cache.get_pos() == seq_len

    # Forward again with single token (autoregressive decoding)
    idx_next = torch.randint(0, model.config.vocab_size, (batch_size, 1), device=device)
    model.forward(idx_next, kv_cache=kv_cache)
    assert kv_cache.get_pos() == seq_len + 1


def test_kv_cache_autoregressive_decoding(model, kv_cache, device):
    """Test multiple forward passes with KV cache (autoregressive decoding)."""
    model.eval()
    batch_size = 1
    vocab_size = model.config.vocab_size

    # Initial prefill
    seq_len = 5
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits1 = model.forward(idx, kv_cache=kv_cache)

    # Autoregressive decoding - single token at a time
    for i in range(3):
        idx_next = torch.randint(0, vocab_size, (batch_size, 1), device=device)
        logits2 = model.forward(idx_next, kv_cache=kv_cache)
        assert logits2.shape == (batch_size, 1, vocab_size)

    assert kv_cache.get_pos() == seq_len + 3


def test_sequence_length_within_rotary_cache(model, device):
    """Test sequence length within rotary cache limit."""
    model.eval()
    # Rotary cache is sequence_len * 10
    max_seq_len = model.config.sequence_len * 10
    batch_size = 1

    # Should work for sequence length up to max_seq_len
    idx = torch.randint(0, model.config.vocab_size, (batch_size, max_seq_len), device=device)
    logits = model.forward(idx)
    assert logits.shape == (batch_size, max_seq_len, model.config.vocab_size)


# ============================================================================
# Component Tests
# ============================================================================

def test_causal_self_attention_forward(small_config, device):
    """Test CausalSelfAttention forward pass."""
    attn = CausalSelfAttention(small_config, layer_idx=0)
    attn.to(device)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, small_config.n_embd, device=device)

    # Create rotary embeddings
    head_dim = small_config.n_embd // small_config.n_head
    cos = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    sin = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    cos_sin = (cos, sin)

    y = attn(x, cos_sin, kv_cache=None)

    assert y.shape == (batch_size, seq_len, small_config.n_embd)


def test_causal_self_attention_with_gqa(gqa_config, device):
    """Test CausalSelfAttention with Group Query Attention."""
    attn = CausalSelfAttention(gqa_config, layer_idx=0)
    attn.to(device)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, gqa_config.n_embd, device=device)

    # Create rotary embeddings
    head_dim = gqa_config.n_embd // gqa_config.n_head
    cos = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    sin = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    cos_sin = (cos, sin)

    y = attn(x, cos_sin, kv_cache=None)

    assert y.shape == (batch_size, seq_len, gqa_config.n_embd)


def test_mlp_forward(small_config, device):
    """Test MLP forward pass."""
    mlp = MLP(small_config)
    mlp.to(device)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, small_config.n_embd, device=device)

    y = mlp(x)

    assert y.shape == (batch_size, seq_len, small_config.n_embd)


def test_block_forward(small_config, device):
    """Test Block forward pass."""
    block = Block(small_config, layer_idx=0)
    block.to(device)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, small_config.n_embd, device=device)

    # Create rotary embeddings
    head_dim = small_config.n_embd // small_config.n_head
    cos = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    sin = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    cos_sin = (cos, sin)

    y = block(x, cos_sin, kv_cache=None)

    assert y.shape == (batch_size, seq_len, small_config.n_embd)


def test_apply_rotary_emb(device):
    """Test rotary embeddings application."""
    batch_size = 2
    num_heads = 4
    seq_len = 10
    head_dim = 16

    # apply_rotary_emb expects (B, T, H, D) format, not (B, H, T, D)
    # So we need to reshape or create in the correct format
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    cos = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    sin = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)

    y = apply_rotary_emb(x, cos, sin)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_qk_norm(small_config, device):
    """Test QK norm application."""
    # QK norm is applied in CausalSelfAttention
    attn = CausalSelfAttention(small_config, layer_idx=0)
    attn.to(device)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, small_config.n_embd, device=device)

    # Create rotary embeddings
    head_dim = small_config.n_embd // small_config.n_head
    cos = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    sin = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    cos_sin = (cos, sin)

    # Get q, k before norm
    q = attn.c_q(x).view(batch_size, seq_len, small_config.n_head, head_dim)
    k = attn.c_k(x).view(batch_size, seq_len, small_config.n_kv_head, head_dim)

    # Apply rotary and norm (as done in forward)
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q_normed = norm(q)
    k_normed = norm(k)

    # Check that norm is applied (values should be normalized)
    # RMS norm normalizes by RMS, so RMS of normalized tensor should be close to 1
    # Compute RMS along the last dimension
    q_rms = torch.sqrt(torch.mean(q_normed ** 2, dim=-1))
    k_rms = torch.sqrt(torch.mean(k_normed ** 2, dim=-1))
    # Should be approximately 1 (with some tolerance for numerical precision)
    assert torch.allclose(q_rms, torch.ones_like(q_rms), atol=0.1)
    assert torch.allclose(k_rms, torch.ones_like(k_rms), atol=0.1)


# ============================================================================
# Optimizer Setup Tests
# ============================================================================

def test_setup_optimizers_creates_optimizers(model):
    """Test setup_optimizers creates correct optimizers."""
    optimizers = model.setup_optimizers()

    assert len(optimizers) == 2
    # First should be AdamW, second should be Muon
    # We can't easily check the exact type without importing, but we can check structure
    assert hasattr(optimizers[0], 'param_groups')
    assert hasattr(optimizers[1], 'param_groups')


def test_setup_optimizers_parameter_groups(model):
    """Test parameter grouping (matrix, embedding, lm_head)."""
    optimizers = model.setup_optimizers()
    adamw_opt, muon_opt = optimizers

    # Check AdamW has 2 groups (lm_head and embedding)
    assert len(adamw_opt.param_groups) == 2

    # Muon groups parameters by their numel (number of elements), not by type
    # So it may have multiple groups if parameters have different sizes
    assert len(muon_opt.param_groups) >= 1

    # Verify all parameters are accounted for
    total_params_in_optimizers = sum(
        sum(p.numel() for p in group['params'])
        for opt in optimizers
        for group in opt.param_groups
    )
    total_params_in_model = sum(p.numel() for p in model.parameters())
    assert total_params_in_optimizers == total_params_in_model


def test_setup_optimizers_lr_scaling(model):
    """Test learning rate scaling based on model dimension."""
    # Test with default model (n_embd=64)
    optimizers1 = model.setup_optimizers(unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02)

    # Create a model with different dimension
    config_large = GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,  # Different dimension
    )
    model_large = GPT(config_large)
    model_large.to(model.get_device())
    model_large.init_weights()
    optimizers2 = model_large.setup_optimizers(unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02)

    # Check that learning rates are different (scaled by dimension)
    adamw1 = optimizers1[0]
    adamw2 = optimizers2[0]

    # LRs should be scaled by (n_embd / 768) ** -0.5
    # For n_embd=64: scale = (64/768)**-0.5 ≈ 3.464
    # For n_embd=128: scale = (128/768)**-0.5 ≈ 2.449
    # So the second should have different (but both non-zero) LRs
    lr1 = adamw1.param_groups[0]['lr']
    lr2 = adamw2.param_groups[0]['lr']
    assert lr1 != lr2
    assert lr1 > 0
    assert lr2 > 0


# ============================================================================
# Utility Method Tests
# ============================================================================

def test_estimate_flops(model):
    """Test estimate_flops returns reasonable value."""
    flops = model.estimate_flops()

    assert isinstance(flops, int) or isinstance(flops, float)
    assert flops > 0
    # Should be a large number for a reasonable model
    assert flops > 1000


def test_rotary_embedding_precomputation(model, device):
    """Test rotary embedding precomputation."""
    head_dim = model.config.n_embd // model.config.n_head
    seq_len = model.rotary_seq_len

    cos, sin = model._precompute_rotary_embeddings(seq_len, head_dim, device=device)

    # Check shape
    assert cos.shape == (1, seq_len, 1, head_dim // 2)
    assert sin.shape == (1, seq_len, 1, head_dim // 2)

    # Check dtype
    assert cos.dtype == torch.bfloat16
    assert sin.dtype == torch.bfloat16

    # Check values are in reasonable range (cos and sin should be in [-1, 1])
    assert torch.all(cos >= -1.0)
    assert torch.all(cos <= 1.0)
    assert torch.all(sin >= -1.0)
    assert torch.all(sin <= 1.0)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_sequence_length_exceeds_rotary_cache(model, device):
    """Test sequence length exceeding rotary cache limit raises assertion."""
    model.eval()
    batch_size = 1
    # Try to use sequence length beyond rotary cache
    max_seq_len = model.config.sequence_len * 10
    seq_len = max_seq_len + 1

    idx = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    with pytest.raises(AssertionError, match="Sequence length grew beyond"):
        model.forward(idx)


def test_device_mismatch_rotary_embeddings(model, device):
    """Test device mismatch between input and rotary embeddings raises assertion."""
    model.eval()
    batch_size = 1
    seq_len = 5

    # Create input on different device (if available)
    if torch.cuda.is_available() and device.type == "cpu":
        other_device = torch.device("cuda")
        idx = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=other_device)

        with pytest.raises(AssertionError, match="different devices"):
            model.forward(idx)
    else:
        # Skip if we can't test device mismatch
        pytest.skip("Cannot test device mismatch - only one device available")


def test_invalid_config_n_embd_not_divisible_by_n_head():
    """Test invalid config (n_embd not divisible by n_head) raises assertion."""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=2,
        n_head=3,  # 3 doesn't divide 64
        n_kv_head=3,
        n_embd=64,
    )

    with pytest.raises(AssertionError):
        CausalSelfAttention(config, layer_idx=0)


def test_invalid_config_n_head_not_divisible_by_n_kv_head():
    """Test invalid config (n_head not divisible by n_kv_head) raises assertion."""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=2,
        n_head=5,  # 5 is not divisible by 3
        n_kv_head=3,
        n_embd=64,
    )

    with pytest.raises(AssertionError):
        CausalSelfAttention(config, layer_idx=0)


def test_invalid_config_n_kv_head_greater_than_n_head():
    """Test invalid config (n_kv_head > n_head) raises assertion."""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_kv_head=4,  # 4 > 2
        n_embd=64,
    )

    with pytest.raises(AssertionError):
        CausalSelfAttention(config, layer_idx=0)


# ============================================================================
# Model Comparison Tests
# ============================================================================

def _copy_model_weights(source_model, target_model):
    """Copy weights from source model to target model."""
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    # Only copy weights that exist in both models
    for key in source_state:
        if key in target_state and source_state[key].shape == target_state[key].shape:
            target_state[key].copy_(source_state[key])

    target_model.load_state_dict(target_state, strict=False)


def test_model_comparison_forward_inference(model, alternative_model, sample_input, device):
    """Compare forward pass outputs in inference mode between reference and alternative model."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    # Copy weights from reference to alternative for fair comparison
    _copy_model_weights(model, alternative_model)

    model.eval()
    alternative_model.eval()

    with torch.no_grad():
        logits_ref = model.forward(sample_input)
        logits_alt = alternative_model.forward(sample_input)

    # Check shapes match
    assert logits_ref.shape == logits_alt.shape, \
        f"Shape mismatch: reference {logits_ref.shape} vs alternative {logits_alt.shape}"

    # Check values are close (allowing for small numerical differences)
    # Use a reasonable tolerance for floating point comparison
    assert torch.allclose(logits_ref, logits_alt, atol=1e-5, rtol=1e-5), \
        f"Logits differ: max diff = {torch.abs(logits_ref - logits_alt).max().item():.2e}"


def test_model_comparison_forward_training(model, alternative_model, sample_input, sample_targets, device):
    """Compare forward pass outputs in training mode (loss) between reference and alternative model."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    # Copy weights from reference to alternative for fair comparison
    _copy_model_weights(model, alternative_model)

    model.train()
    alternative_model.train()

    loss_ref = model.forward(sample_input, targets=sample_targets)
    loss_alt = alternative_model.forward(sample_input, targets=sample_targets)

    # Check loss values are close
    assert torch.allclose(loss_ref, loss_alt, atol=1e-5, rtol=1e-5), \
        f"Loss differs: reference {loss_ref.item():.6f} vs alternative {loss_alt.item():.6f}"


def test_model_comparison_generation(model, alternative_model, device):
    """Compare generation outputs between reference and alternative model."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    # Copy weights from reference to alternative for fair comparison
    _copy_model_weights(model, alternative_model)

    tokens = [1, 2, 3]
    max_tokens = 5
    seed = 42
    temperature = 0.0  # Use greedy decoding for deterministic comparison

    model.eval()
    alternative_model.eval()

    with torch.no_grad():
        generated_ref = list(model.generate(tokens, max_tokens=max_tokens, temperature=temperature, seed=seed))
        generated_alt = list(alternative_model.generate(tokens, max_tokens=max_tokens, temperature=temperature, seed=seed))

    # Check generated tokens match
    assert generated_ref == generated_alt, \
        f"Generation differs: reference {generated_ref} vs alternative {generated_alt}"


def test_model_comparison_kv_cache(model, alternative_model, kv_cache, device):
    """Compare KV cache behavior between reference and alternative model."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    # Copy weights from reference to alternative for fair comparison
    _copy_model_weights(model, alternative_model)

    model.eval()
    alternative_model.eval()

    batch_size = 1
    seq_len = 5
    idx = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    # Create separate KV caches for each model
    kv_cache_ref = KVCache(batch_size, model.config.n_kv_head, model.config.sequence_len,
                           model.config.n_embd // model.config.n_head, model.config.n_layer)
    kv_cache_alt = KVCache(batch_size, alternative_model.config.n_kv_head, alternative_model.config.sequence_len,
                           alternative_model.config.n_embd // alternative_model.config.n_head, alternative_model.config.n_layer)

    with torch.no_grad():
        logits_ref = model.forward(idx, kv_cache=kv_cache_ref)
        logits_alt = alternative_model.forward(idx, kv_cache=kv_cache_alt)

    # Check outputs match
    assert torch.allclose(logits_ref, logits_alt, atol=1e-5, rtol=1e-5), \
        f"KV cache outputs differ: max diff = {torch.abs(logits_ref - logits_alt).max().item():.2e}"

    # Check KV cache positions match
    assert kv_cache_ref.get_pos() == kv_cache_alt.get_pos(), \
        f"KV cache positions differ: reference {kv_cache_ref.get_pos()} vs alternative {kv_cache_alt.get_pos()}"


def test_model_comparison_parameter_shapes(model, alternative_model):
    """Compare parameter shapes between reference and alternative model."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    ref_params = dict(model.named_parameters())
    alt_params = dict(alternative_model.named_parameters())

    # Check that all reference parameters exist in alternative model
    missing_params = []
    shape_mismatches = []

    for name, param in ref_params.items():
        if name not in alt_params:
            missing_params.append(name)
        elif param.shape != alt_params[name].shape:
            shape_mismatches.append((name, param.shape, alt_params[name].shape))

    if missing_params:
        pytest.fail(f"Alternative model missing parameters: {missing_params}")

    if shape_mismatches:
        mismatch_str = "\n".join([f"  {name}: ref {ref_shape} vs alt {alt_shape}"
                                  for name, ref_shape, alt_shape in shape_mismatches])
        pytest.fail(f"Parameter shape mismatches:\n{mismatch_str}")


def test_model_comparison_different_inputs(model, alternative_model, small_config, device):
    """Compare models with various input shapes."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    # Copy weights from reference to alternative for fair comparison
    _copy_model_weights(model, alternative_model)

    model.eval()
    alternative_model.eval()

    test_cases = [
        (1, 1),   # Single token, single batch
        (2, 5),   # Small batch, short sequence
        (4, 20),  # Larger batch, longer sequence
    ]

    for batch_size, seq_len in test_cases:
        idx = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            logits_ref = model.forward(idx)
            logits_alt = alternative_model.forward(idx)

        assert torch.allclose(logits_ref, logits_alt, atol=1e-5, rtol=1e-5), (
            f"Outputs differ for batch_size={batch_size}, seq_len={seq_len}: "
            f"max diff = {torch.abs(logits_ref - logits_alt).max().item():.2e}"
        )


def test_model_comparison_block_components(model, alternative_model, small_config, device):
    """Compare a single transformer block between reference and alternative model."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    # Ensure weights are aligned between the two models
    _copy_model_weights(model, alternative_model)

    model.eval()
    alternative_model.eval()

    # Basic structural assumptions (same components as reference model)
    assert hasattr(alternative_model, "transformer")
    assert hasattr(alternative_model.transformer, "h")
    assert len(alternative_model.transformer.h) == len(model.transformer.h) == small_config.n_layer

    # Use a small synthetic input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, small_config.n_embd, device=device)

    # Create synthetic rotary embeddings (same shape for both blocks)
    head_dim = small_config.n_embd // small_config.n_head
    cos = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    sin = torch.randn(1, seq_len, 1, head_dim // 2, dtype=torch.bfloat16, device=device)
    cos_sin = (cos, sin)

    ref_block = model.transformer.h[0]
    alt_block = alternative_model.transformer.h[0]

    with torch.no_grad():
        y_ref = ref_block(x, cos_sin, kv_cache=None)
        y_alt = alt_block(x, cos_sin, kv_cache=None)

    # Check shapes and numerical closeness of block outputs
    assert y_ref.shape == y_alt.shape == (batch_size, seq_len, small_config.n_embd)
    assert torch.allclose(y_ref, y_alt, atol=1e-5, rtol=1e-5), (
        f"Block outputs differ: max diff = {torch.abs(y_ref - y_alt).max().item():.2e}"
    )


def test_model_comparison_causal_self_attention_forward(
    model, alternative_model, small_config, device
):
    """Compare CausalSelfAttention forward between reference and alternative model."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    _copy_model_weights(model, alternative_model)

    model.eval()
    alternative_model.eval()

    assert hasattr(model.transformer.h[0], "attn")
    assert hasattr(alternative_model.transformer.h[0], "attn")

    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, small_config.n_embd, device=device)

    head_dim = small_config.n_embd // small_config.n_head
    cos = torch.randn(
        1,
        seq_len,
        1,
        head_dim // 2,
        dtype=torch.bfloat16,
        device=device,
    )
    sin = torch.randn(
        1,
        seq_len,
        1,
        head_dim // 2,
        dtype=torch.bfloat16,
        device=device,
    )
    cos_sin = (cos, sin)

    ref_attn = model.transformer.h[0].attn
    alt_attn = alternative_model.transformer.h[0].attn

    with torch.no_grad():
        y_ref = ref_attn(x, cos_sin, kv_cache=None)
        y_alt = alt_attn(x, cos_sin, kv_cache=None)

    assert y_ref.shape == y_alt.shape == (batch_size, seq_len, small_config.n_embd)
    assert torch.allclose(y_ref, y_alt, atol=1e-5, rtol=1e-5), (
        "CausalSelfAttention outputs differ: max diff = "
        f"{torch.abs(y_ref - y_alt).max().item():.2e}"
    )


def test_model_comparison_causal_self_attention_with_gqa(
    alternative_model_class, gqa_config, device
):
    """Compare CausalSelfAttention with GQA between reference and alternative model."""
    if alternative_model_class is None:
        pytest.skip("No alternative model provided")

    model_gqa = GPT(gqa_config)
    model_gqa.to(device)
    model_gqa.init_weights()

    try:
        alt_model_gqa = alternative_model_class(gqa_config)
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"Could not create alternative GQA model instance: {exc}")

    alt_model_gqa.to(device)
    if hasattr(alt_model_gqa, "init_weights"):
        alt_model_gqa.init_weights()

    _copy_model_weights(model_gqa, alt_model_gqa)

    model_gqa.eval()
    alt_model_gqa.eval()

    assert hasattr(model_gqa.transformer.h[0], "attn")
    assert hasattr(alt_model_gqa.transformer.h[0], "attn")

    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, gqa_config.n_embd, device=device)

    head_dim = gqa_config.n_embd // gqa_config.n_head
    cos = torch.randn(
        1,
        seq_len,
        1,
        head_dim // 2,
        dtype=torch.bfloat16,
        device=device,
    )
    sin = torch.randn(
        1,
        seq_len,
        1,
        head_dim // 2,
        dtype=torch.bfloat16,
        device=device,
    )
    cos_sin = (cos, sin)

    ref_attn = model_gqa.transformer.h[0].attn
    alt_attn = alt_model_gqa.transformer.h[0].attn

    with torch.no_grad():
        y_ref = ref_attn(x, cos_sin, kv_cache=None)
        y_alt = alt_attn(x, cos_sin, kv_cache=None)

    assert y_ref.shape == y_alt.shape == (batch_size, seq_len, gqa_config.n_embd)
    assert torch.allclose(y_ref, y_alt, atol=1e-5, rtol=1e-5), (
        "GQA CausalSelfAttention outputs differ: max diff = "
        f"{torch.abs(y_ref - y_alt).max().item():.2e}"
    )


def test_model_comparison_mlp_components(model, alternative_model, small_config, device):
    """Compare MLP component between reference and alternative model."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    _copy_model_weights(model, alternative_model)

    model.eval()
    alternative_model.eval()

    assert hasattr(model.transformer.h[0], "mlp")
    assert hasattr(alternative_model.transformer.h[0], "mlp")

    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, small_config.n_embd, device=device)

    ref_mlp = model.transformer.h[0].mlp
    alt_mlp = alternative_model.transformer.h[0].mlp

    with torch.no_grad():
        y_ref = ref_mlp(x)
        y_alt = alt_mlp(x)

    assert y_ref.shape == y_alt.shape == (batch_size, seq_len, small_config.n_embd)
    assert torch.allclose(y_ref, y_alt, atol=1e-5, rtol=1e-5), (
        "MLP outputs differ: max diff = "
        f"{torch.abs(y_ref - y_alt).max().item():.2e}"
    )


def test_model_comparison_rotary_and_qk_norm(model, alternative_model, small_config, device):
    """Compare rotary embeddings and QK norm behavior between models."""
    if alternative_model is None:
        pytest.skip("No alternative model provided")

    _copy_model_weights(model, alternative_model)

    model.eval()
    alternative_model.eval()

    assert hasattr(model.transformer.h[0], "attn")
    assert hasattr(alternative_model.transformer.h[0], "attn")

    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, small_config.n_embd, device=device)

    head_dim = small_config.n_embd // small_config.n_head
    cos = torch.randn(
        1,
        seq_len,
        1,
        head_dim // 2,
        dtype=torch.bfloat16,
        device=device,
    )
    sin = torch.randn(
        1,
        seq_len,
        1,
        head_dim // 2,
        dtype=torch.bfloat16,
        device=device,
    )

    ref_attn = model.transformer.h[0].attn
    alt_attn = alternative_model.transformer.h[0].attn

    with torch.no_grad():
        q_ref = ref_attn.c_q(x).view(
            batch_size,
            seq_len,
            small_config.n_head,
            head_dim,
        )
        k_ref = ref_attn.c_k(x).view(
            batch_size,
            seq_len,
            small_config.n_kv_head,
            head_dim,
        )
        q_alt = alt_attn.c_q(x).view(
            batch_size,
            seq_len,
            small_config.n_head,
            head_dim,
        )
        k_alt = alt_attn.c_k(x).view(
            batch_size,
            seq_len,
            small_config.n_kv_head,
            head_dim,
        )

        q_ref_rot = apply_rotary_emb(q_ref, cos, sin)
        k_ref_rot = apply_rotary_emb(k_ref, cos, sin)
        q_alt_rot = apply_rotary_emb(q_alt, cos, sin)
        k_alt_rot = apply_rotary_emb(k_alt, cos, sin)

        q_ref_norm = norm(q_ref_rot)
        k_ref_norm = norm(k_ref_rot)
        q_alt_norm = norm(q_alt_rot)
        k_alt_norm = norm(k_alt_rot)

    # Rotary embedding comparison
    assert torch.allclose(q_ref_rot, q_alt_rot, atol=1e-5, rtol=1e-5), (
        "Rotary Q projections differ: max diff = "
        f"{torch.abs(q_ref_rot - q_alt_rot).max().item():.2e}"
    )
    assert torch.allclose(k_ref_rot, k_alt_rot, atol=1e-5, rtol=1e-5), (
        "Rotary K projections differ: max diff = "
        f"{torch.abs(k_ref_rot - k_alt_rot).max().item():.2e}"
    )

    # QK norm comparison
    assert torch.allclose(q_ref_norm, q_alt_norm, atol=1e-5, rtol=1e-5), (
        "Q normed projections differ: max diff = "
        f"{torch.abs(q_ref_norm - q_alt_norm).max().item():.2e}"
    )
    assert torch.allclose(k_ref_norm, k_alt_norm, atol=1e-5, rtol=1e-5), (
        "K normed projections differ: max diff = "
        f"{torch.abs(k_ref_norm - k_alt_norm).max().item():.2e}"
    )
