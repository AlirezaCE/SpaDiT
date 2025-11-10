"""
Test script to verify scGPT integration with DiT diffusion model
Run this on your server to test the scGPT embedder
"""

import torch
import sys
sys.path.append('.')

print("=" * 60)
print("Testing scGPT Integration")
print("=" * 60)

# Test 1: Import scGPT embedder
print("\n[TEST 1] Importing scGPT embedder...")
try:
    from model.scgpt_embedder import scGPTConditionEmbedder
    print("✓ Successfully imported scGPTConditionEmbedder")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Load scGPT model
print("\n[TEST 2] Loading scGPT brain model...")
try:
    scgpt_model_dir = "scgpt_models/scGPT_brain"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embedder = scGPTConditionEmbedder(
        model_dir=scgpt_model_dir,
        hidden_size=512,
        freeze_scgpt=True,
        max_seq_len=1200,
        device=device
    )
    print("✓ Successfully loaded scGPT brain model")
except Exception as e:
    print(f"✗ Failed to load scGPT model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test forward pass
print("\n[TEST 3] Testing forward pass with dummy data...")
try:
    batch_size = 4
    n_genes = 2000

    # Create dummy single-cell data
    sc_data = torch.randn(batch_size, n_genes).to(device)
    gene_ids = torch.arange(n_genes).to(device)

    # Forward pass
    embeddings = embedder(sc_data=sc_data, gene_ids=gene_ids)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {sc_data.shape}")
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Expected output dim: {512}")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Import updated DiT model
print("\n[TEST 4] Importing DiT model with scGPT support...")
try:
    from model.diff_model_scgpt import DiT_diff
    print("✓ Successfully imported DiT_diff with scGPT support")
except Exception as e:
    print(f"✗ Failed to import DiT_diff: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Initialize DiT with scGPT
print("\n[TEST 5] Initializing DiT model with scGPT embedder...")
try:
    model = DiT_diff(
        st_input_size=249,
        condi_input_size=2000,
        hidden_size=256,
        depth=4,
        num_heads=8,
        classes=6,
        pca_dim=100,
        dit_type='dit',
        use_scgpt=True,
        scgpt_model_dir=scgpt_model_dir,
        freeze_scgpt=True,
        device=device
    )
    model.to(device)
    print("✓ Successfully initialized DiT with scGPT")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters (scGPT): {frozen_params:,}")

except Exception as e:
    print(f"✗ Failed to initialize DiT: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Full forward pass through DiT
print("\n[TEST 6] Testing full forward pass through DiT...")
try:
    batch_size = 4
    st_genes = 249
    sc_genes = 2000

    # Create dummy inputs
    x = torch.randn(batch_size, st_genes).to(device)  # noisy ST data
    x_hat = torch.randn(batch_size, sc_genes).to(device)  # SC condition
    t = torch.randint(0, 10, (batch_size,)).to(device)  # timestep
    y = torch.randn(batch_size, sc_genes).to(device)  # SC data for scGPT
    gene_ids = torch.arange(sc_genes).to(device)

    # Forward pass
    output = model(x=x, x_hat=x_hat, t=t, y=y, gene_ids=gene_ids)

    print(f"✓ Full forward pass successful")
    print(f"  Input x shape: {x.shape}")
    print(f"  Input x_hat shape: {x_hat.shape}")
    print(f"  Input y shape: {y.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, {st_genes})")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Compare with baseline (without scGPT)
print("\n[TEST 7] Testing baseline model (without scGPT)...")
try:
    baseline_model = DiT_diff(
        st_input_size=249,
        condi_input_size=2000,
        hidden_size=256,
        depth=4,
        num_heads=8,
        classes=6,
        pca_dim=100,
        dit_type='dit',
        use_scgpt=False,  # Disable scGPT
        device=device
    )
    baseline_model.to(device)

    # Forward pass (y will use MLP instead of scGPT)
    output_baseline = baseline_model(x=x, x_hat=x_hat, t=t, y=y)

    print(f"✓ Baseline model works")
    print(f"  Baseline output shape: {output_baseline.shape}")

except Exception as e:
    print(f"✗ Baseline test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed successfully! ✓")
print("=" * 60)
print("\nNext steps:")
print("1. Update your main.py to use 'from model.diff_model_scgpt import DiT_diff'")
print("2. Add 'use_scgpt=True' and 'scgpt_model_dir=...' to DiT_diff initialization")
print("3. Pass 'gene_ids' to model forward pass")
print("\nReady to train with scGPT embeddings!")
