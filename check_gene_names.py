"""
Quick diagnostic script to check gene names in dataset
"""
import scanpy as sc
import json

# Load the scGPT vocabulary
with open('scgpt_models/scGPT_brain/vocab.json', 'r') as f:
    vocab = json.load(f)

# Remove special tokens
vocab_genes = set([k for k in vocab.keys() if not k.startswith('<')])
print(f"scGPT vocab has {len(vocab_genes)} genes (excluding special tokens)")
print(f"First 10 genes in vocab: {list(vocab_genes)[:10]}")

# Try to find the dataset files
import os
import glob

# Look for h5ad files
h5ad_files = glob.glob('dataset_ML/**/*.h5ad', recursive=True)
if not h5ad_files:
    h5ad_files = glob.glob('**/*.h5ad', recursive=True)

print(f"\nFound {len(h5ad_files)} h5ad files:")
for f in h5ad_files:
    print(f"  - {f}")

if h5ad_files:
    # Check the first file
    print(f"\n=== Checking first file: {h5ad_files[0]} ===")
    adata = sc.read_h5ad(h5ad_files[0])

    data_genes = adata.var_names.tolist()
    print(f"Dataset has {len(data_genes)} genes")
    print(f"First 20 genes in dataset: {data_genes[:20]}")

    # Check overlap
    data_genes_set = set(data_genes)
    overlap = vocab_genes & data_genes_set
    print(f"\nGene overlap: {len(overlap)} / {len(data_genes)} ({len(overlap)/len(data_genes)*100:.2f}%)")

    if overlap:
        print(f"Example matching genes: {list(overlap)[:10]}")

    # Check if genes might be Ensembl IDs
    if data_genes[0].startswith('ENSG'):
        print("\n⚠️  Your genes are Ensembl IDs (e.g., ENSG00000...), but scGPT vocab uses gene symbols")
    elif data_genes[0].upper() != data_genes[0]:
        print("\n⚠️  Your genes might have different capitalization")

    # Try case-insensitive matching
    vocab_genes_lower = {g.lower(): g for g in vocab_genes}
    data_genes_lower = {g.lower(): g for g in data_genes}
    case_insensitive_overlap = set(vocab_genes_lower.keys()) & set(data_genes_lower.keys())

    if len(case_insensitive_overlap) > len(overlap):
        print(f"\n✓ Case-insensitive matching finds {len(case_insensitive_overlap)} genes ({len(case_insensitive_overlap)/len(data_genes)*100:.2f}%)")
