"""
Test if the dataloader is working and how long iteration takes
"""
import torch
import time
from torch.utils.data import DataLoader
from preprocess.dataset import ConditionalDiffusionDataset, split_dataset_with_gene_names
from model.scgpt_wrapper import create_gene_id_mapping
import scanpy as sc
from scgpt.tokenizer import GeneVocab
import os

print("Loading dataset...")

# Paths
sc_path = 'datasets/dataset_ML/sc/dataset_ML_sc.h5ad'
st_path = 'datasets/dataset_ML/st/dataset_ML_st.h5ad'

# Load scGPT vocabulary and create gene mappings
vocab_path = 'scgpt_models/scGPT_brain/vocab.json'
vocab = GeneVocab.from_file(vocab_path)

sc_adata = sc.read_h5ad(sc_path)
st_adata = sc.read_h5ad(st_path)

sc_gene_ids, st_gene_ids = create_gene_id_mapping(sc_adata, st_adata, vocab)
gene_ids = sc_gene_ids

# Create dataset
dataset = ConditionalDiffusionDataset(sc_path, st_path, gene_ids=gene_ids)
print(f"Dataset size: {len(dataset)}")

# Split dataset
(train_dataset, train_gene_names), _, _ = split_dataset_with_gene_names(
    dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42
)

print(f"Train dataset size: {len(train_dataset)}")

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
print(f"Number of batches: {len(train_dataloader)}")

print("\nTesting dataloader iteration...")
for i, batch_data in enumerate(train_dataloader):
    start = time.time()

    if len(batch_data) == 4:
        x, x_hat, x_cond, gene_ids_batch = batch_data
        print(f"Batch {i}:")
        print(f"  x shape: {x.shape}")
        print(f"  x_hat shape: {x_hat.shape}")
        print(f"  x_cond shape: {x_cond.shape}")
        print(f"  gene_ids shape: {gene_ids_batch.shape}")
    else:
        x, x_hat, x_cond = batch_data
        print(f"Batch {i}:")
        print(f"  x shape: {x.shape}")
        print(f"  x_hat shape: {x_hat.shape}")
        print(f"  x_cond shape: {x_cond.shape}")

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.4f}s\n")

    if i >= 2:  # Test first 3 batches
        break

print("✓ Dataloader test complete!")
