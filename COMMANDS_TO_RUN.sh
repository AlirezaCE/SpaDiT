#!/bin/bash
# ============================================================================
# Commands to run on your server to test scGPT integration
# ============================================================================

echo "============================================"
echo "  scGPT Integration Test Commands"
echo "============================================"
echo ""

# Navigate to SpaDiT directory
cd /path/to/SpaDiT  # UPDATE THIS PATH!

# Step 1: Check Python and dependencies
echo "[Step 1] Checking environment..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Step 2: Check scGPT installation
echo "[Step 2] Checking scGPT installation..."
python -c "import scgpt; print('scGPT installed ✓')" || {
    echo "scGPT not found. Installing..."
    pip install scgpt
}
echo ""

# Step 3: Check files exist
echo "[Step 3] Checking required files..."
echo "Checking model files:"
ls -lh model/scgpt_embedder.py
ls -lh model/diff_model_scgpt.py
ls -lh test_scgpt_integration.py

echo ""
echo "Checking scGPT brain model:"
ls -lh scgpt_models/scGPT_brain/args.json
ls -lh scgpt_models/scGPT_brain/best_model.pt
ls -lh scgpt_models/scGPT_brain/vocab.json
echo ""

# Step 4: Run the test
echo "[Step 4] Running integration test..."
echo "============================================"
python test_scgpt_integration.py

# Step 5: Check GPU memory
echo ""
echo "[Step 5] GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "============================================"
echo "  Test Complete!"
echo "============================================"
echo ""
echo "If all tests passed, update main.py:"
echo "  1. Change: from model.diff_model import DiT_diff"
echo "     To:     from model.diff_model_scgpt import DiT_diff"
echo ""
echo "  2. Add to model initialization:"
echo "     use_scgpt=True,"
echo "     scgpt_model_dir='scgpt_models/scGPT_brain',"
echo "     freeze_scgpt=True,"
echo "     device=args.device"
echo ""
