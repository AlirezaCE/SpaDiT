# scGPT Integration for SpaDiT

This guide shows how to integrate scGPT pretrained embeddings into your DiT diffusion model for spatial transcriptomics.

## What Changed

Instead of using a simple `nn.Embedding` for condition embeddings, we now use the pretrained **scGPT brain model** to generate rich biological embeddings from single-cell gene expression data.

## Files Added/Modified

1. **`model/scgpt_embedder.py`** - New scGPT wrapper for condition embedding
2. **`model/diff_model_scgpt.py`** - Updated DiT model with scGPT support
3. **`test_scgpt_integration.py`** - Test script to verify integration

## Installation

Make sure you have scGPT dependencies installed on your server:

```bash
pip install scgpt "flash-attn<1.0.5"
# OR if you have issues with flash-attn:
pip install scgpt
```

## Quick Start - Testing

### Step 1: Test the integration

**Run on your server:**
```bash
cd /path/to/SpaDiT
python test_scgpt_integration.py
```

This will:
- Load the scGPT brain model from `scgpt_models/scGPT_brain/`
- Test the embedder with dummy data
- Initialize the DiT model with scGPT
- Run a full forward pass
- Compare with baseline (without scGPT)

Expected output:
```
============================================================
Testing scGPT Integration
============================================================

[TEST 1] Importing scGPT embedder...
✓ Successfully imported scGPTConditionEmbedder

[TEST 2] Loading scGPT brain model...
Using device: cuda
Loading gene vocabulary from scgpt_models/scGPT_brain/vocab.json
Loaded pretrained scGPT weights from scgpt_models/scGPT_brain/best_model.pt
scGPT parameters frozen
✓ Successfully loaded scGPT brain model

[TEST 3] Testing forward pass with dummy data...
✓ Forward pass successful
  Input shape: torch.Size([4, 2000])
  Output shape: torch.Size([4, 512])
  Expected output dim: 512

... (more tests)

All tests completed successfully! ✓
```

### Step 2: Update your training code

There are two options:

#### Option A: Minimal change (recommended)

Update `main.py` to import the new model:

```python
# Change this line:
# from model.diff_model import DiT_diff

# To this:
from model.diff_model_scgpt import DiT_diff

# Then update model initialization:
model = DiT_diff(
    st_input_size=spot_num,
    condi_input_size=cell_num,
    hidden_size=args.hidden_size,
    depth=args.depth,
    num_heads=args.head,
    classes=6,
    mlp_ratio=4.0,
    pca_dim=args.pca_dim,
    dit_type='dit',

    # NEW: Add these parameters
    use_scgpt=True,
    scgpt_model_dir='scgpt_models/scGPT_brain',
    freeze_scgpt=True,  # Keep scGPT frozen during training
    device=args.device
)
```

#### Option B: Make it configurable via argparse

Add to your argparse in `main.py`:

```python
parser.add_argument("--use_scgpt", type=bool, default=True)
parser.add_argument("--scgpt_model_dir", type=str, default='scgpt_models/scGPT_brain')
parser.add_argument("--freeze_scgpt", type=bool, default=True)

# Then in model initialization:
model = DiT_diff(
    ...
    use_scgpt=args.use_scgpt,
    scgpt_model_dir=args.scgpt_model_dir if args.use_scgpt else None,
    freeze_scgpt=args.freeze_scgpt,
    device=args.device
)
```

## Model Architecture

### Without scGPT (original):
```
SC data → SimpleMLP → embedding (512-dim)
                          ↓
                      Time + Condition
                          ↓
                        DiT/UNet
```

### With scGPT (new):
```
SC data → Tokenize → scGPT Transformer → Projection → embedding (512-dim)
                                                           ↓
                                                   Time + Condition
                                                           ↓
                                                         DiT/UNet
```

## Key Features

1. **Pretrained on 13.2M brain cells**: scGPT brain model captures rich biological patterns
2. **Frozen by default**: scGPT parameters are frozen to preserve pretrained knowledge
3. **Automatic tokenization**: Handles gene name to ID mapping internally
4. **Backward compatible**: Can still use the original SimpleMLP by setting `use_scgpt=False`

## How the Embedder Works

The `scGPTConditionEmbedder` does the following:

1. **Tokenization**: Converts gene expression `[batch, genes]` to tokens
2. **scGPT Encoding**: Passes through pretrained transformer
3. **Cell Embedding**: Extracts CLS token representation
4. **Projection**: Maps scGPT dim (512) to your hidden_size (256*2=512)

## Training Tips

### Memory Considerations

scGPT adds ~400M parameters (but frozen, so no gradient storage). If you run out of memory:

1. Reduce batch size
2. Use gradient checkpointing (add to embedder if needed)
3. Precompute embeddings offline (see below)

### Precompute Embeddings (Advanced)

If scGPT is too slow during training, precompute embeddings:

```python
from model.scgpt_embedder import scGPTConditionEmbedder
import torch

# Load embedder
embedder = scGPTConditionEmbedder(
    model_dir='scgpt_models/scGPT_brain',
    hidden_size=512,
    freeze_scgpt=True
)
embedder.eval()

# Precompute for all SC data
with torch.no_grad():
    sc_embeddings = embedder(sc_data, gene_ids)

# Save embeddings
torch.save(sc_embeddings, 'precomputed_scgpt_embeddings.pt')

# Then in training, use the simpler embedder:
from model.scgpt_embedder import scGPTConditionEmbedderSimple
simple_embedder = scGPTConditionEmbedderSimple(input_dim=512, hidden_size=512)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'scgpt'"

**Solution**: Install scGPT
```bash
pip install scgpt
```

### Issue: "flash_attn is not installed"

**Solution**: This is just a warning. scGPT will use standard PyTorch attention. To use flash attention:
```bash
# Make sure you have CUDA 11.7+
pip install "flash-attn<1.0.5"
```

### Issue: "FileNotFoundError: vocab.json not found"

**Solution**: Make sure the scGPT brain model is downloaded:
```bash
ls scgpt_models/scGPT_brain/
# Should show: args.json  best_model.pt  vocab.json
```

Download from: https://drive.google.com/drive/folders/1vf1ijfQSk7rGdDGpBntR5bi5g6gNt-Gx

### Issue: CUDA out of memory

**Solutions**:
1. Reduce batch size
2. Use CPU for scGPT (slower but works):
   ```python
   scgpt_embedder = scGPTConditionEmbedder(..., device='cpu')
   ```
3. Precompute embeddings offline

## Performance Comparison

You should see:
- Better reconstruction quality (lower MSE)
- More biologically meaningful embeddings
- Slightly slower training (scGPT forward pass)
- Similar or better downstream metrics

## Command Examples

### Train with scGPT (default):
```bash
python main.py \
    --document dataset_ML \
    --batch_size 64 \
    --hidden_size 256 \
    --epoch 20 \
    --device cuda:0
```

### Train without scGPT (baseline):
```bash
python main.py \
    --document dataset_ML \
    --batch_size 64 \
    --hidden_size 256 \
    --epoch 20 \
    --device cuda:0 \
    --use_scgpt False
```

### Train with unfrozen scGPT (fine-tuning):
```bash
python main.py \
    --document dataset_ML \
    --batch_size 32 \  # Smaller batch due to gradients
    --hidden_size 256 \
    --epoch 20 \
    --device cuda:0 \
    --freeze_scgpt False
```

## Next Steps

1. **Run the test script** to verify everything works
2. **Update main.py** with the new import and parameters
3. **Train a model** and compare with baseline
4. **Adjust hyperparameters** if needed
5. **Try other scGPT models** (whole-human, etc.) by changing `scgpt_model_dir`

## Questions?

Check:
- scGPT docs: https://scgpt.readthedocs.io/
- scGPT paper: https://www.biorxiv.org/content/10.1101/2023.04.30.538439
- Model files in `model/scgpt_embedder.py` for implementation details
