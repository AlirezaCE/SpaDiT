# Quick Start - Run This On Your Server

## Step 1: Upload files to server

Make sure these NEW files are on your server:
```
SpaDiT/
├── model/
│   ├── scgpt_embedder.py          [NEW]
│   └── diff_model_scgpt.py        [NEW]
├── test_scgpt_integration.py      [NEW]
└── scgpt_models/
    └── scGPT_brain/
        ├── args.json
        ├── best_model.pt
        └── vocab.json
```

## Step 2: Test the integration

Run this command on your server:

```bash
python test_scgpt_integration.py
```

**Send me the full output from this command.**

Expected output should look like:
```
============================================================
Testing scGPT Integration
============================================================

[TEST 1] Importing scGPT embedder...
✓ Successfully imported scGPTConditionEmbedder

[TEST 2] Loading scGPT brain model...
...
```

## Step 3: If test passes, update main.py

Replace this line in `main.py`:
```python
from model.diff_model import DiT_diff
```

With:
```python
from model.diff_model_scgpt import DiT_diff
```

And update the model initialization (around line 113) to:
```python
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
    use_scgpt=True,                              # ADD THIS
    scgpt_model_dir='scgpt_models/scGPT_brain',  # ADD THIS
    freeze_scgpt=True,                           # ADD THIS
    device=args.device                           # ADD THIS
)
```

## Step 4: Train your model

```bash
python main.py \
    --document dataset_ML \
    --batch_size 64 \
    --hidden_size 256 \
    --epoch 20 \
    --device cuda:0
```

## What to Send Me

Please run the test script and send me:
1. **The complete terminal output** from `python test_scgpt_integration.py`
2. Any error messages if it fails
3. Your GPU memory available: `nvidia-smi`

## Common Issues

### If you see "ModuleNotFoundError: No module named 'scgpt'"
```bash
pip install scgpt
```

### If you see issues with scGPT path
The test script expects:
- scGPT repo in: `SpaDiT/scGPT/`
- Model files in: `SpaDiT/scgpt_models/scGPT_brain/`

### If CUDA out of memory
Try smaller batch size or use CPU for scGPT:
```python
use_scgpt=True,
scgpt_model_dir='scgpt_models/scGPT_brain',
freeze_scgpt=True,
device='cpu'  # Use CPU for scGPT
```

---

**PLEASE RUN: `python test_scgpt_integration.py` and send me the output!**
