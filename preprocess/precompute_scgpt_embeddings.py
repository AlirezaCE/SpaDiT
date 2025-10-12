"""
Pre-compute scGPT embeddings for single cell data
Run this script on the server after installing scGPT

Usage:
    python preprocess/precompute_scgpt_embeddings.py --document dataset_ML --device cuda:0 --model_path /path/to/scGPT_brain
"""

import argparse
import os
import sys
import numpy as np
import scanpy as sc
import torch
import json
import pickle
from pathlib import Path

# Import scGPT modules
try:
    import scgpt
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.preprocess import Preprocessor
    from scgpt.utils import set_seed
except ImportError as e:
    print(f"Error importing scgpt: {e}")
    print("Please install: pip install scgpt ipython")
    sys.exit(1)

parser = argparse.ArgumentParser(description='Pre-compute scGPT embeddings for single cell data')
parser.add_argument('--document', type=str, default='dataset_ML', help='Dataset name')
parser.add_argument('--sc_data', type=str, default='_sc.h5ad', help='Single cell data filename')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained scGPT model directory')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding computation')
parser.add_argument('--max_seq_len', type=int, default=1200, help='Maximum sequence length for scGPT')
args = parser.parse_args()


def load_pretrained_model(model_dir, vocab, device):
    """
    Load pre-trained scGPT model from checkpoint directory
    """
    print(f"Loading scGPT model from {model_dir}...")

    model_dir = Path(model_dir)

    # Find model file
    possible_files = ["model.pt", "best_model.pt", "pytorch_model.bin"]
    model_file = None
    for fname in possible_files:
        fpath = model_dir / fname
        if fpath.exists():
            model_file = fpath
            break

    if model_file is None:
        print(f"Error: No model file found in {model_dir}")
        print("Directory contents:")
        for f in model_dir.iterdir():
            print(f"  {f.name}")
        raise FileNotFoundError(f"Model file not found in {model_dir}")

    print(f"Found model checkpoint: {model_file}")

    # Try to load config
    config_file = model_dir / "args.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        embsize = config.get('embsize', 512)
        nhead = config.get('nheads', 8)
        d_hid = config.get('d_hid', 512)
        nlayers = config.get('nlayers', 12)
        n_bins = config.get('n_bins', 51)
        print(f"Loaded config from {config_file}")
    else:
        # Default brain model parameters
        embsize = 512
        nhead = 8
        d_hid = 512
        nlayers = 12
        n_bins = 51
        print("Using default brain model configuration")

    print(f"Model config: embsize={embsize}, nhead={nhead}, nlayers={nlayers}")

    # Create model
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        vocab=vocab,
        dropout=0.0,
        pad_token="<pad>",
        pad_value=0,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        n_input_bins=n_bins,
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
    )

    # Load weights
    try:
        checkpoint = torch.load(model_file, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise

    model.to(device)
    model.eval()
    return model, embsize


def tokenize_and_pad_batch(batch_data, gene_ids, max_len):
    """
    Tokenize and pad a batch of cells
    """
    batch_tokens = []
    batch_values = []

    for cell_data in batch_data:
        if hasattr(cell_data, 'toarray'):
            cell_data = cell_data.toarray().flatten()
        else:
            cell_data = np.array(cell_data).flatten()

        # Get non-zero genes
        nonzero_idx = np.nonzero(cell_data)[0]

        # Limit to max_len
        if len(nonzero_idx) > max_len:
            # Sort by expression value and keep top max_len
            top_idx = np.argsort(cell_data[nonzero_idx])[-max_len:]
            nonzero_idx = nonzero_idx[top_idx]

        tokens = gene_ids[nonzero_idx]
        values = cell_data[nonzero_idx]

        batch_tokens.append(tokens)
        batch_values.append(values)

    # Pad sequences
    max_batch_len = min(max(len(t) for t in batch_tokens), max_len)

    tokens_padded = np.zeros((len(batch_tokens), max_batch_len), dtype=np.int64)
    values_padded = np.zeros((len(batch_tokens), max_batch_len), dtype=np.float32)

    for i, (tokens, values) in enumerate(zip(batch_tokens, batch_values)):
        length = min(len(tokens), max_batch_len)
        tokens_padded[i, :length] = tokens[:length]
        values_padded[i, :length] = values[:length]

    return tokens_padded, values_padded


def extract_embeddings(model, adata, vocab, device, batch_size, max_len):
    """
    Extract cell embeddings using the scGPT model
    """
    print("Extracting scGPT embeddings...")

    n_cells = adata.n_obs
    embeddings_list = []

    # Create gene ID mapping
    gene_names = adata.var_names.tolist()
    gene_to_id = {gene: idx for idx, gene in enumerate(gene_names)}
    gene_ids = np.arange(len(gene_names))

    # Process in batches
    for i in range(0, n_cells, batch_size):
        batch_end = min(i + batch_size, n_cells)
        print(f"Processing cells {i} to {batch_end}/{n_cells}")

        # Get batch
        batch_data = adata.X[i:batch_end]

        # Tokenize and pad
        tokens, values = tokenize_and_pad_batch(batch_data, gene_ids, max_len)

        # Convert to tensors
        tokens_tensor = torch.from_numpy(tokens).long().to(device)
        values_tensor = torch.from_numpy(values).float().to(device)

        # Create padding mask
        src_key_padding_mask = tokens_tensor.eq(vocab["<pad>"])

        # Get embeddings
        with torch.no_grad():
            try:
                # Use encode_batch method if available
                cell_emb = model.encode_batch(
                    tokens_tensor,
                    values_tensor,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    return_np=True
                )
            except AttributeError:
                # Fallback: use forward pass and extract CLS token
                output = model(
                    tokens_tensor,
                    values_tensor,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None
                )
                # Extract CLS token embedding (first token)
                if isinstance(output, dict) and 'cell_emb' in output:
                    cell_emb = output['cell_emb'].cpu().numpy()
                elif isinstance(output, tuple):
                    cell_emb = output[0][:, 0, :].cpu().numpy()
                else:
                    cell_emb = output[:, 0, :].cpu().numpy()

        embeddings_list.append(cell_emb)

    # Concatenate all batches
    embeddings = np.vstack(embeddings_list)
    print(f"Final embeddings shape: {embeddings.shape}")

    return embeddings


def main():
    print("="*60)
    print("Pre-computing scGPT embeddings for SpaDiT")
    print("="*60)

    # Set seed for reproducibility
    set_seed(42)

    # Setup paths
    sc_path = f'datasets/{args.document}/sc/{args.document}{args.sc_data}'
    output_dir = f'datasets/{args.document}/scgpt_embeddings/'
    os.makedirs(output_dir, exist_ok=True)

    output_emb_path = os.path.join(output_dir, f'{args.document}_scgpt_embeddings.npy')
    output_vocab_path = os.path.join(output_dir, f'{args.document}_gene_vocab.pkl')

    # Load single cell data
    print(f"\nLoading single cell data from: {sc_path}")
    adata = sc.read_h5ad(sc_path)
    print(f"Data shape: {adata.shape} (cells x genes)")
    print(f"Memory usage: {adata.X.data.nbytes / 1e6:.2f} MB" if hasattr(adata.X, 'data') else '')

    # Load vocab from model directory or create from data
    model_vocab_path = Path(args.model_path) / "vocab.json"
    if model_vocab_path.exists():
        print(f"\nLoading vocabulary from model: {model_vocab_path}")
        vocab = GeneVocab.from_file(model_vocab_path)
    else:
        print("\nCreating vocabulary from data...")
        gene_names = adata.var_names.tolist()
        vocab = GeneVocab.from_dict({gene: idx for idx, gene in enumerate(gene_names)})

    # Save vocab
    with open(output_vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Saved gene vocabulary to: {output_vocab_path}")

    # Preprocess data
    print("\nPreprocessing data for scGPT...")
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=False,
        subset_hvg=False,
        binning=51,
        result_binned_key="X_binned",
    )
    adata_processed = preprocessor(adata, batch_key=None)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, embsize = load_pretrained_model(args.model_path, vocab, device)

    # Extract embeddings
    embeddings = extract_embeddings(
        model,
        adata_processed,
        vocab,
        device,
        args.batch_size,
        args.max_seq_len
    )

    # Save embeddings
    np.save(output_emb_path, embeddings)
    print(f"\nSaved embeddings to: {output_emb_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings size: {embeddings.nbytes / 1e6:.2f} MB")

    print("="*60)
    print("Done! You can now train SpaDiT with --use_scgpt flag")
    print("="*60)


if __name__ == '__main__':
    main()
