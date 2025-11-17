# Implementation Summary: scGPT Integration (Option 2)

## What Was Implemented

You wanted to replace the simple `nn.Embedding(classes, hidden_size)` with **scGPT embeddings** that provide dynamic, biologically-informed gene embeddings based on expression patterns.

### ✅ Option 2: Dynamic Gene Embeddings for Conditioning

Instead of fixed embeddings, we now:
1. Take your scRNA-seq gene expression data (e.g., 10 genes with their values)
2. Pass it through frozen scGPT model
3. Get a **unique 512-dim embedding** for that specific expression pattern
4. Use this as conditioning for the diffusion model

**Key insight**: Each different gene expression pattern gets its own unique embedding!

## Files Created

### 1. `model/scgpt_wrapper.py`
**Purpose**: Wrapper around scGPT for easy integration

**Key classes**:
- `scGPTEmbedder`: Loads pre-trained scGPT brain model and generates embeddings
- `create_gene_id_mapping()`: Maps gene names to vocabulary indices

**Usage**:
```python
embedder = scGPTEmbedder('scgpt_models/scGPT_brain')
cell_emb = embedder(gene_ids, expression_values)  # (batch, 512)
```

### 2. `main_with_scgpt.py`
**Purpose**: Complete training script with scGPT integration

**Features**:
- Automatic gene vocabulary loading
- Gene ID mapping
- Training with scGPT embeddings
- Comparison with baseline

### 3. `README_scGPT.md`
**Purpose**: Documentation and usage guide

### 4. `test_scgpt_integration.py`
**Purpose**: Verify the integration works

**Run this first**:
```bash
python test_scgpt_integration.py
```

## Files Modified

### 1. `model/diff_model.py`
**Changes**:
- Added `use_scgpt` and `scgpt_model_dir` parameters to `DiT_diff.__init__()`
- Integrated `scGPTEmbedder`
- Added `scgpt_projection` layer (512 → hidden_size*2)
- Updated `forward()` to accept `gene_ids` and use scGPT embeddings
- **Changed to use DiT blocks instead of UNet** (UNet code commented out)

**Key code**:
```python
if self.use_scgpt and self.scgpt_embedder is not None:
    scgpt_emb = self.scgpt_embedder(gene_ids, y)  # (batch, 512)
    y = self.scgpt_projection(scgpt_emb)  # Project to hidden_size*2
```

### 2. `preprocess/data.py`
**Changes**:
- Added `gene_ids` parameter to `ConditionalDiffusionDataset`
- Updated `__getitem__` to return gene IDs
- Added `get_sc_gene_names()` method

### 3. `model/diff_train.py`
**Changes**:
- Updated training loop to handle datasets with/without gene IDs
- Pass `gene_ids` to model forward pass

### 4. `model/sample.py`
**Changes**:
- Updated `model_sample_diff()` to handle gene IDs
- Pass gene IDs during inference

## How It Works: Data Flow

### Training Time

```
1. Load Data
   ├─ scRNA-seq: (n_cells, n_sc_genes)
   └─ ST: (n_spots, n_st_genes)

2. Gene Vocabulary Mapping
   ├─ Load scGPT vocab (60k genes)
   ├─ Map SC genes → vocab indices
   └─ Store gene_ids: array of shape (n_sc_genes,)

3. For Each Batch
   ├─ SC expression: (batch, n_sc_genes)
   ├─ gene_ids: (n_sc_genes,)  # same for all samples
   └─ Pass to scGPT:
       ├─ Tokenize (filter non-zero, add <cls>)
       ├─ scGPT forward (frozen)
       └─ Extract cell embedding: (batch, 512)

4. Conditioning
   ├─ Project: (batch, 512) → (batch, hidden_size*2)
   ├─ Add timestep embedding
   └─ Pass through DiT blocks

5. Loss & Backprop
   └─ Only train: DiT blocks + projection layer
       (scGPT frozen)
```

### Inference Time

Same process as training, but:
- Model in eval mode
- Iterative denoising with scGPT conditioning at each step

## Key Features

### 1. Dynamic Embeddings
```python
# Sample 1: high expression of NEUROD1, GAD1
expr1 = [10.5, 8.2, 0.1, ...]
emb1 = scGPT(gene_ids, expr1)  # Unique embedding

# Sample 2: different expression pattern
expr2 = [0.3, 0.5, 9.8, ...]
emb2 = scGPT(gene_ids, expr2)  # Different embedding!
```

### 2. Frozen scGPT
- scGPT weights are **never updated**
- Only train: DiT blocks + projection layer
- Prevents catastrophic forgetting

### 3. Biological Context
- scGPT trained on millions of brain cells
- Captures gene co-expression patterns
- Understands biological relationships

### 4. Easy Toggle
```bash
# With scGPT
python main_with_scgpt.py --use_scgpt

# Without scGPT (baseline)
python main_with_scgpt.py
```

## Pre-trained Model

**Location**: `scgpt_models/scGPT_brain/`

**Specs**:
- Vocab: 60,697 genes
- Embedding: 512 dims
- Layers: 12 transformer layers
- Heads: 8 attention heads
- Training: Brain tissue from CELLxGENE

**Files**:
- `vocab.json`: Gene vocabulary
- `best_model.pt`: Pre-trained weights (frozen)
- `args.json`: Model configuration

## Usage Example

### Quick Test
```bash
# 1. Test integration
python test_scgpt_integration.py

# 2. Train with scGPT
python main_with_scgpt.py \
    --use_scgpt \
    --document dataset_ML \
    --epoch 20 \
    --batch_size 64
```

### Compare Results
```bash
# Baseline (no scGPT)
python main_with_scgpt.py --document dataset_ML

# With scGPT
python main_with_scgpt.py --use_scgpt --document dataset_ML

# Results saved with different names:
# - SpaDiT_prediction.csv (baseline)
# - SpaDiT_scgpt_prediction.csv (with scGPT)
```

## Benefits vs Original

| Aspect | Original | With scGPT |
|--------|----------|------------|
| Embedding | Fixed random | Dynamic, context-aware |
| Biological context | None | Pre-trained on brain data |
| Gene relationships | Not captured | Co-expression learned |
| Parameters | Train from scratch | Transfer learning |
| Conditioning | Simple MLP | scGPT transformer |

## Potential Improvements

1. **Fine-tune scGPT**: Currently frozen, could fine-tune last few layers
2. **Use gene-level embeddings**: Currently using cell-level, could use per-gene
3. **Multi-modal**: Combine scGPT with other embeddings
4. **Attention**: Use cross-attention between scGPT and DiT

## Troubleshooting

### Gene Coverage Low?
Check which genes are in vocab:
```python
from scgpt.tokenizer import GeneVocab
vocab = GeneVocab.from_file('scgpt_models/scGPT_brain/vocab.json')
print([g for g in your_genes if g not in vocab])
```

### Out of Memory?
- Reduce `--batch_size`
- Reduce `--depth` (number of DiT blocks)

### scGPT not loading?
- Check path: `scgpt_models/scGPT_brain/`
- Verify files exist: `vocab.json`, `best_model.pt`

## Next Steps

1. **Run test**: `python test_scgpt_integration.py`
2. **Train model**: `python main_with_scgpt.py --use_scgpt`
3. **Compare results**: Check `result/` directory
4. **Tune hyperparameters**: Learning rate, batch size, etc.

## Questions?

The key difference from Option 1 (which would use gene-level embeddings) is that Option 2 uses **cell-level embeddings** - we feed the entire gene expression profile through scGPT and get one embedding vector that represents the whole cell/sample's transcriptional state. This is more suitable for conditioning a diffusion model because it provides a compact, biologically meaningful representation of the input condition.
