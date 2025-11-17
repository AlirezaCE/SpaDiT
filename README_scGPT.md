# SpaDiT with scGPT Embeddings

## Overview

This implementation integrates scGPT pre-trained gene embeddings into SpaDiT's diffusion model for spatial gene expression prediction. Instead of using simple random embeddings for conditioning, we now use **dynamic, context-aware embeddings** from scGPT that capture biological gene expression patterns.

## What Changed?

### Option 2 Implementation: Dynamic Gene Embeddings for Conditioning

- **Before**: Used `nn.Embedding(classes, hidden_size)` - fixed random embeddings
- **After**: Use scGPT's transformer to generate **dynamic embeddings** based on actual gene expression patterns

For each sample with specific gene expression values, scGPT produces a unique 512-dimensional embedding that captures the biological context, which is then projected to your hidden dimension.

## Architecture

```
scRNA-seq data (batch, n_genes)
    ↓
Gene IDs (n_genes,) ────┐
    ↓                   │
Expression values  ──→  scGPT Model (frozen)
                        ↓
                   Cell Embedding (batch, 512)
                        ↓
                   Projection Layer
                        ↓
                   Conditioning (batch, hidden_size*2)
                        ↓
                   Combined with timestep
                        ↓
                   DiT Blocks (16 layers)
                        ↓
                   Predicted ST expression
```

## Files Modified

1. **`model/scgpt_wrapper.py`** (NEW)
   - `scGPTEmbedder`: Loads pre-trained scGPT brain model
   - `create_gene_id_mapping()`: Maps gene names to vocab indices

2. **`model/diff_model.py`**
   - Added `use_scgpt` and `scgpt_model_dir` parameters
   - Integrated scGPT embedder in `__init__()`
   - Updated `forward()` to use scGPT embeddings
   - DiT blocks now used (UNet commented out)

3. **`preprocess/data.py`**
   - Updated `ConditionalDiffusionDataset` to handle gene IDs

4. **`model/diff_train.py`**
   - Updated training loop to pass gene IDs

5. **`model/sample.py`**
   - Updated sampling to pass gene IDs

6. **`main_with_scgpt.py`** (NEW)
   - Complete training script with scGPT integration

## Usage

### Option 1: Run with scGPT (Recommended)

```bash
python main_with_scgpt.py \
    --document dataset_ML \
    --use_scgpt \
    --scgpt_model_dir scgpt_models/scGPT_brain \
    --device cuda:0 \
    --epoch 20 \
    --batch_size 64 \
    --hidden_size 256 \
    --depth 16 \
    --head 16
```

### Option 2: Run without scGPT (Original)

```bash
python main_with_scgpt.py \
    --document dataset_ML \
    --device cuda:0 \
    --epoch 20 \
    --batch_size 64
```

## Requirements

Make sure you have scGPT installed:

```bash
# scGPT is already in your scGPT/ directory
# No additional installation needed if paths are correct
```

## Pre-trained Model

Your scGPT brain model is located at:
```
scgpt_models/scGPT_brain/
├── vocab.json           # Gene vocabulary (60k+ genes)
├── best_model.pt        # Pre-trained weights
└── args.json           # Model configuration
```

**Model specs:**
- Embedding size: 512
- Layers: 12
- Attention heads: 8
- Max sequence length: 1200
- Trained on brain tissue data

## How It Works

### 1. Gene Vocabulary Mapping

```python
# Load scGPT vocabulary
vocab = GeneVocab.from_file('scgpt_models/scGPT_brain/vocab.json')

# Map your genes to vocab indices
# Example: "NEUROD1" → 12345, "GAD1" → 23456
gene_ids = np.array([vocab[gene] for gene in your_genes])
```

### 2. Dynamic Embedding Generation

For each batch during training:

```python
# Input: gene_ids (n_genes,), expression (batch, n_genes)
# Output: embeddings (batch, 512)

scgpt_emb = scgpt_embedder(gene_ids, expression_values)
# Each sample gets a UNIQUE embedding based on its expression pattern
```

### 3. Conditioning the Diffusion Model

```python
# Project scGPT embeddings to your dimension
conditioning = projection(scgpt_emb)  # (batch, 512) → (batch, 512)

# Combine with timestep
c = timestep_emb + conditioning

# Pass through DiT blocks
for block in dit_blocks:
    x = block(x, c)
```

## Benefits

1. **Biological Context**: scGPT embeddings capture co-expression patterns learned from millions of cells
2. **Dynamic Embeddings**: Each expression pattern gets a unique embedding (not fixed)
3. **Pre-trained Knowledge**: Leverages scGPT's brain-specific training
4. **Frozen Weights**: scGPT weights are frozen, only training your DiT model

## Gene Coverage

When you run the script, it will report gene coverage:

```
Gene coverage: 87.3% (1234/1414 genes in vocab)
```

Genes not in vocabulary are mapped to `<pad>` token.

## Output

Results are saved with suffix `_scgpt`:

```
result/dataset_ML/
├── SpaDiT_scgpt_prediction.csv    # Predictions with scGPT
├── original_scgpt.csv              # Ground truth
└── ...
```

Compare with baseline (without scGPT):
```
result/dataset_ML/
├── SpaDiT_prediction.csv          # Original predictions
├── original.csv                    # Ground truth
└── ...
```

## Troubleshooting

### Issue: Gene coverage too low

**Solution**: Your dataset genes might not overlap well with scGPT's brain vocabulary. Check which genes are missing:

```python
from scgpt.tokenizer import GeneVocab
vocab = GeneVocab.from_file('scgpt_models/scGPT_brain/vocab.json')

missing_genes = [g for g in your_genes if g not in vocab]
print(f"Missing genes: {missing_genes}")
```

### Issue: Out of memory

**Solution**: Reduce batch size or use gradient checkpointing:

```bash
python main_with_scgpt.py --batch_size 32 --use_scgpt
```

### Issue: scGPT not loading

**Solution**: Check that flash_attn is disabled in config (already set to `False` in args.json)

## Parameters

Key parameters for scGPT:

- `--use_scgpt`: Enable scGPT embeddings
- `--scgpt_model_dir`: Path to pre-trained model (default: `scgpt_models/scGPT_brain`)
- `--hidden_size`: Your model's hidden dimension (default: 256)
- `--depth`: Number of DiT blocks (default: 16)

## Performance Tips

1. **Start with frozen scGPT**: Always keep scGPT frozen (already set)
2. **Adjust projection layer**: The `scgpt_projection` layer is trainable and maps 512 → hidden_size*2
3. **Learning rate**: May need to tune LR when using scGPT (default: 1e-4)
4. **Gene coverage**: Higher coverage = better performance

## Citation

If you use this implementation, please cite both:

1. **SpaDiT**: Li et al. (2024)
2. **scGPT**: Cui et al. (2024)

## Questions?

The key insight: scGPT provides **dynamic embeddings** for each expression pattern, not fixed embeddings. This captures the biological context of which genes are expressed together.
