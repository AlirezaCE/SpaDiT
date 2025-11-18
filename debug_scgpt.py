"""
Debug script to test where scGPT is hanging
"""
import torch
import numpy as np
import sys
import os
import time

# Add scGPT to path
scgpt_path = os.path.join(os.path.dirname(__file__), 'scGPT')
if scgpt_path not in sys.path:
    sys.path.insert(0, scgpt_path)

from scgpt.tokenizer import GeneVocab, tokenize_and_pad_batch

print("Loading vocabulary...")
vocab = GeneVocab.from_file('scgpt_models/scGPT_brain/vocab.json')
print(f"Vocab loaded: {len(vocab)} genes")

# Create dummy data similar to what training would use
batch_size = 64
n_genes = 2279

# Simulate gene_ids (these would be the mapped vocab indices)
gene_ids = np.random.randint(0, len(vocab), size=n_genes)
print(f"Gene IDs shape: {gene_ids.shape}")

# Simulate expression values (some zeros, some non-zero)
expression_values = np.random.rand(batch_size, n_genes).astype(np.float32)
expression_values[expression_values < 0.7] = 0  # Make it sparse like real data
print(f"Expression values shape: {expression_values.shape}")
print(f"Non-zero rate: {(expression_values > 0).mean():.2%}")

print("\nTesting tokenize_and_pad_batch...")
start = time.time()

try:
    tokenized = tokenize_and_pad_batch(
        data=expression_values,
        gene_ids=gene_ids,
        max_len=1200,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=0,
        append_cls=True,
        include_zero_gene=False,
        cls_token="<cls>",
        return_pt=True,
    )
    elapsed = time.time() - start
    print(f"✓ Tokenization completed in {elapsed:.2f} seconds")
    print(f"  genes shape: {tokenized['genes'].shape}")
    print(f"  values shape: {tokenized['values'].shape}")

except Exception as e:
    elapsed = time.time() - start
    print(f"✗ Tokenization FAILED after {elapsed:.2f} seconds")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing full scGPT embedder forward pass...")
print("="*60)

from model.scgpt_wrapper import scGPTEmbedder

print("Initializing scGPT embedder...")
start = time.time()
embedder = scGPTEmbedder(
    scgpt_model_dir='scgpt_models/scGPT_brain',
    freeze_scgpt=True
)
embedder = embedder.to('cuda:0')
elapsed = time.time() - start
print(f"✓ Embedder initialized in {elapsed:.2f} seconds")

# Convert to tensors on GPU
gene_ids_tensor = torch.from_numpy(gene_ids).long().to('cuda:0')
expression_tensor = torch.from_numpy(expression_values).float().to('cuda:0')

print(f"\nInput shapes:")
print(f"  gene_ids: {gene_ids_tensor.shape}, device: {gene_ids_tensor.device}")
print(f"  expression: {expression_tensor.shape}, device: {expression_tensor.device}")

print("\nRunning forward pass...")
start = time.time()
try:
    with torch.no_grad():
        output = embedder(gene_ids_tensor, expression_tensor)
    elapsed = time.time() - start
    print(f"✓ Forward pass completed in {elapsed:.2f} seconds")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ Forward pass FAILED after {elapsed:.2f} seconds")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting multiple forward passes (simulating training)...")
for i in range(5):
    start = time.time()
    with torch.no_grad():
        output = embedder(gene_ids_tensor, expression_tensor)
    elapsed = time.time() - start
    print(f"  Batch {i+1}: {elapsed:.2f}s")
