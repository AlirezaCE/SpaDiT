# Quick Start: SpaDiT with scGPT

## 🚀 Get Started in 3 Steps

### Step 1: Test the Integration (1 minute)

```bash
python test_scgpt_integration.py
```

**Expected output**:
```
================================================================================
Testing scGPT Integration
================================================================================

[Test 1] Importing scGPT wrapper...
✓ Successfully imported scGPT wrapper

[Test 2] Loading scGPT brain model...
✓ Successfully loaded scGPT model
   - Vocabulary size: 60697
   - Embedding dimension: 512
   - Layers: 12
   - Attention heads: 8

[Test 3] Testing embedding generation...
✓ Successfully generated embeddings
   - Input shape: torch.Size([4, 100])
   - Output shape: torch.Size([4, 512])
   ✓ Shape verification passed

[Test 4] Testing DiT model integration...
✓ Successfully created DiT model with scGPT
✓ Forward pass successful
✓ Output shape verification passed

[Test 5] Testing dataset with gene IDs...
✓ Dataset class supports gene_ids parameter

================================================================================
All tests passed! ✓
================================================================================
```

---

### Step 2: Train Your Model (minutes to hours)

```bash
# Train WITH scGPT embeddings
python main_with_scgpt.py \
    --use_scgpt \
    --document dataset_ML \
    --device cuda:0 \
    --epoch 20 \
    --batch_size 64

# OR train WITHOUT scGPT (baseline)
python main_with_scgpt.py \
    --document dataset_ML \
    --device cuda:0 \
    --epoch 20 \
    --batch_size 64
```

**What happens**:
1. ✅ Loads scGPT brain model vocabulary
2. ✅ Maps your genes to vocabulary indices
3. ✅ Reports gene coverage (e.g., "87% genes in vocab")
4. ✅ Creates DiT model with scGPT embedder
5. ✅ Trains model (scGPT frozen, only DiT trained)
6. ✅ Runs inference and saves results

---

### Step 3: Check Results

```bash
# Results saved in:
result/dataset_ML/
├── SpaDiT_scgpt_prediction.csv     # WITH scGPT
├── original_scgpt.csv
├── SpaDiT_prediction.csv           # Baseline
└── original.csv
```

---

## 📊 What You'll See During Training

```
=== Loading scGPT vocabulary and creating gene mappings ===
Loading scGPT vocabulary from scgpt_models/scGPT_brain/vocab.json
Loaded scGPT vocabulary with 60697 genes
SC gene coverage: 87.30% (1234/1414 genes in vocab)
ST gene coverage: 92.15% (986/1070 genes in vocab)

Dataset info:
  SC genes: 1414, spots/cells: 249
  ST genes: 1070, spots: 249
  Train: 747, Valid: 214, Test: 109

Initializing scGPT embedder for conditioning...
Loading scGPT vocabulary from scgpt_models/scGPT_brain/vocab.json
scGPT config: d_model=512, nhead=8, nlayers=12
Loading pre-trained scGPT weights from scgpt_models/scGPT_brain/best_model.pt
Loaded weights - Missing: 0, Unexpected: 0
scGPT model frozen for inference
scGPT embedder initialized. Will project 512 -> 512

=== Training model with scGPT ===
noise loss:0.00123, lr:1.00e-04: 100%|██████████| 20/20
Model saved to save/dataset_ML_ckpt/dataset_ML_scdiff/dataset_ML_scgpt.pt

=== Running inference ===
time: 9: 100%|██████████| 10/10 [PCC:0.85432, RMSE:1.23456]
```

---

## 🎯 Key Differences: With vs Without scGPT

| Feature | Without scGPT | With scGPT |
|---------|--------------|------------|
| **Conditioning** | Simple MLP on expression values | scGPT transformer embeddings |
| **Embedding** | Random, trained from scratch | Pre-trained, biologically informed |
| **Gene context** | No co-expression knowledge | Learned from millions of cells |
| **Training** | All parameters trained | scGPT frozen, only DiT trained |
| **File saved** | `model.pt` | `model_scgpt.pt` |

---

## ⚙️ Command Line Options

### Basic Options
```bash
--document dataset_ML          # Dataset name
--device cuda:0                # GPU device
--epoch 20                     # Training epochs
--batch_size 64               # Batch size
```

### scGPT Options
```bash
--use_scgpt                    # Enable scGPT (flag)
--scgpt_model_dir path/to/model  # scGPT model path (default: scgpt_models/scGPT_brain)
```

### Model Architecture
```bash
--hidden_size 256              # Hidden dimension
--depth 16                     # Number of DiT blocks
--head 16                      # Attention heads
--diffusion_step 10           # Diffusion timesteps
```

### Training
```bash
--learning_rate 1e-4          # Learning rate
--mask_nonzero_ratio 0.3      # Masking ratio for non-zero values
--mask_zero_ratio 0.1         # Masking ratio for zero values
--seed 3407                   # Random seed
```

---

## 🔍 Understanding the Output

### Gene Coverage Report
```
SC gene coverage: 87.30% (1234/1414 genes in vocab)
```
- **87.30%**: Percentage of your genes found in scGPT vocab
- **1234/1414**: 1234 genes matched, 1414 total genes
- Higher is better (>80% is good)

### Training Progress
```
noise loss:0.00123, lr:1.00e-04
```
- **noise loss**: Prediction loss (lower is better)
- **lr**: Current learning rate

### Inference Metrics
```
PCC:0.85432, RMSE:1.23456
```
- **PCC**: Pearson Correlation Coefficient (higher is better, 0-1)
- **RMSE**: Root Mean Square Error (lower is better)

---

## 💡 Tips

### Start Small
```bash
# Quick test with fewer epochs
python main_with_scgpt.py --use_scgpt --epoch 5 --batch_size 32
```

### GPU Memory Issues?
```bash
# Reduce batch size
python main_with_scgpt.py --use_scgpt --batch_size 32

# Or reduce model depth
python main_with_scgpt.py --use_scgpt --depth 8
```

### Compare Models
```bash
# 1. Train baseline
python main_with_scgpt.py --document dataset_ML --epoch 20

# 2. Train with scGPT
python main_with_scgpt.py --use_scgpt --document dataset_ML --epoch 20

# 3. Compare results
python -c "
import pandas as pd
baseline = pd.read_csv('result/dataset_ML/SpaDiT_prediction.csv')
scgpt = pd.read_csv('result/dataset_ML/SpaDiT_scgpt_prediction.csv')
print('Correlation:', baseline.corrwith(scgpt).mean())
"
```

---

## 📚 More Information

- **Full documentation**: See `README_scGPT.md`
- **Implementation details**: See `IMPLEMENTATION_SUMMARY.md`
- **Troubleshooting**: See `README_scGPT.md` → Troubleshooting section

---

## ✅ Checklist

Before training, make sure:

- [ ] Test script passes: `python test_scgpt_integration.py`
- [ ] scGPT model exists: `scgpt_models/scGPT_brain/best_model.pt`
- [ ] Data exists: `datasets/dataset_ML/sc/` and `datasets/dataset_ML/st/`
- [ ] GPU available (optional but recommended)

---

## 🎉 You're Ready!

```bash
python main_with_scgpt.py --use_scgpt --document dataset_ML
```

The model will automatically:
1. ✅ Load scGPT pre-trained weights
2. ✅ Map your genes to scGPT vocabulary
3. ✅ Generate dynamic embeddings for each sample
4. ✅ Train the diffusion model with biological conditioning
5. ✅ Save results with `_scgpt` suffix for comparison

Enjoy your biologically-informed spatial gene prediction! 🧬🚀
