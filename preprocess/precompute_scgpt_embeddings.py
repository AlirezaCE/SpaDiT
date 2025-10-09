"""
Pre-compute scGPT embeddings for single cell data
Run this script on the server after installing scGPT

Usage:
    python preprocess/precompute_scgpt_embeddings.py --document dataset_ML --device cuda:0
"""

import argparse
import os
import numpy as np
import scanpy as sc
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
import pickle

parser = argparse.ArgumentParser(description='Pre-compute scGPT embeddings for single cell data')
parser.add_argument('--document', type=str, default='dataset_ML', help='Dataset name')
parser.add_argument('--sc_data', type=str, default='_sc.h5ad', help='Single cell data filename')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--model_path', type=str, default='scgpt_model', help='Path to pre-trained scGPT model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for embedding computation')
parser.add_argument('--max_seq_len', type=int, default=1200, help='Maximum sequence length for scGPT')
parser.add_argument('--embsize', type=int, default=512, help='Embedding size from scGPT')
args = parser.parse_args()

def load_scgpt_model(model_path, device):
    """
    Load pre-trained scGPT model
    You may need to adjust this based on the specific scGPT checkpoint you're using
    """
    print(f"Loading scGPT model from {model_path}...")

    # Load model configuration and weights
    # Adjust these parameters based on your pre-trained model
    model = TransformerModel(
        ntoken=args.max_seq_len,
        d_model=args.embsize,
        nhead=8,
        d_hid=args.embsize,
        nlayers=12,
        vocab=None,  # Will be set later
        dropout=0.0,
        pad_token="<pad>",
        pad_value=0,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        n_input_bins=51,
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=True,
    )

    # Load pre-trained weights if available
    if os.path.exists(model_path):
        model_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_dict)
        print("Loaded pre-trained weights")
    else:
        print(f"Warning: Model path {model_path} not found. Using random initialization.")
        print("Please download pre-trained scGPT model from: https://github.com/bowang-lab/scGPT")

    model.to(device)
    model.eval()
    return model

def preprocess_for_scgpt(adata, gene_vocab=None):
    """
    Preprocess AnnData for scGPT input
    """
    print("Preprocessing data for scGPT...")

    # Basic preprocessing
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,  # Already filtered
        filter_cell_by_counts=False,   # Already filtered
        normalize_total=False,          # Already normalized
        log1p=False,                    # Already log-transformed
        subset_hvg=False,               # Already selected HVG
        hvg_flavor="seurat_v3",
        binning=51,                     # Number of bins for expression values
        result_binned_key="X_binned",
    )

    preprocessor(adata, batch_key=None)
    return adata

def get_gene_vocab(adata):
    """
    Create gene vocabulary from AnnData
    """
    print("Creating gene vocabulary...")
    gene_names = adata.var_names.tolist()
    vocab = GeneVocab.from_dict({'gene_names': gene_names})
    return vocab

def extract_scgpt_embeddings(model, adata, gene_vocab, device, batch_size):
    """
    Extract cell embeddings from scGPT model
    """
    print("Extracting scGPT embeddings...")

    n_cells = adata.n_obs
    embeddings = []

    # Process in batches
    for i in range(0, n_cells, batch_size):
        batch_end = min(i + batch_size, n_cells)
        print(f"Processing cells {i} to {batch_end}/{n_cells}")

        # Get batch data
        batch_adata = adata[i:batch_end].copy()

        # Prepare input tensors
        # This is a simplified version - adjust based on actual scGPT API
        gene_ids = []
        values = []

        for cell_idx in range(batch_adata.n_obs):
            cell_data = batch_adata.X[cell_idx]
            if hasattr(cell_data, 'toarray'):
                cell_data = cell_data.toarray().flatten()
            else:
                cell_data = np.array(cell_data).flatten()

            # Get non-zero genes
            nonzero_idx = np.nonzero(cell_data)[0]
            gene_ids.append(nonzero_idx)
            values.append(cell_data[nonzero_idx])

        # Pad sequences and convert to tensors
        max_len = min(max(len(g) for g in gene_ids), args.max_seq_len)

        gene_ids_padded = np.zeros((len(gene_ids), max_len), dtype=np.int64)
        values_padded = np.zeros((len(gene_ids), max_len), dtype=np.float32)

        for j, (gids, vals) in enumerate(zip(gene_ids, values)):
            length = min(len(gids), max_len)
            gene_ids_padded[j, :length] = gids[:length]
            values_padded[j, :length] = vals[:length]

        gene_ids_tensor = torch.from_numpy(gene_ids_padded).to(device)
        values_tensor = torch.from_numpy(values_padded).to(device)

        # Get embeddings
        with torch.no_grad():
            # Adjust this based on actual scGPT forward function
            cell_embeddings = model.encode_batch(
                gene_ids_tensor,
                values_tensor,
                src_key_padding_mask=gene_ids_tensor.eq(0),
                batch_labels=None,
                return_np=True
            )

        embeddings.append(cell_embeddings)

    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    print(f"Final embeddings shape: {embeddings.shape}")

    return embeddings

def main():
    print("="*50)
    print("Pre-computing scGPT embeddings")
    print("="*50)

    # Set paths
    sc_path = f'datasets/{args.document}/sc/{args.document}{args.sc_data}'
    output_dir = f'datasets/{args.document}/scgpt_embeddings/'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{args.document}_scgpt_embeddings.npy')
    vocab_path = os.path.join(output_dir, f'{args.document}_gene_vocab.pkl')

    print(f"Loading single cell data from: {sc_path}")
    adata = sc.read_h5ad(sc_path)
    print(f"Data shape: {adata.shape} (cells x genes)")

    # Create gene vocabulary
    gene_vocab = get_gene_vocab(adata)

    # Save gene vocabulary
    with open(vocab_path, 'wb') as f:
        pickle.dump(gene_vocab, f)
    print(f"Saved gene vocabulary to: {vocab_path}")

    # Preprocess data for scGPT
    adata = preprocess_for_scgpt(adata, gene_vocab)

    # Load scGPT model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_scgpt_model(args.model_path, device)

    # Extract embeddings
    embeddings = extract_scgpt_embeddings(
        model, adata, gene_vocab, device, args.batch_size
    )

    # Save embeddings
    np.save(output_path, embeddings)
    print(f"Saved embeddings to: {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")

    print("="*50)
    print("Done!")
    print("="*50)

if __name__ == '__main__':
    main()
