"""
scGPT Wrapper for SpaDiT
Extracts dynamic gene embeddings from scGPT based on gene expression patterns
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import numpy as np
import json

# Add scGPT to path
scgpt_path = os.path.join(os.path.dirname(__file__), '..', 'scGPT')
if scgpt_path not in sys.path:
    sys.path.insert(0, scgpt_path)

from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab, tokenize_and_pad_batch


class scGPTEmbedder(nn.Module):
    """
    Extracts dynamic embeddings for genes using pre-trained scGPT model
    """
    def __init__(
        self,
        scgpt_model_dir: str,
        freeze_scgpt: bool = True,
    ):
        """
        Initialize scGPT embedder

        Args:
            scgpt_model_dir: Path to pre-trained scGPT model directory (e.g., 'scgpt_models/scGPT_brain')
            freeze_scgpt: Whether to freeze scGPT weights (recommended: True)
        """
        super().__init__()

        model_dir = Path(scgpt_model_dir)

        # Load config
        args_file = model_dir / "args.json"
        with open(args_file, 'r') as f:
            self.config = json.load(f)

        # Load vocabulary
        vocab_file = model_dir / "vocab.json"
        print(f"Loading scGPT vocabulary from {vocab_file}")
        self.vocab = GeneVocab.from_file(vocab_file)

        # Extract model parameters
        self.d_model = self.config['embsize']  # 512
        self.nhead = self.config['nheads']  # 8
        self.d_hid = self.config['d_hid']  # 512
        self.nlayers = self.config['nlayers']  # 12
        self.max_seq_len = self.config['max_seq_len']  # 1200
        self.pad_token = self.config['pad_token']
        self.pad_value = self.config['pad_value']

        print(f"scGPT config: d_model={self.d_model}, nhead={self.nhead}, nlayers={self.nlayers}")

        # Initialize scGPT model
        self.scgpt_model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.d_model,
            nhead=self.nhead,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            vocab=self.vocab,
            dropout=0.0,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            input_emb_style=self.config['input_emb_style'],
            cell_emb_style="cls",
            explicit_zero_prob=False,
            use_fast_transformer=self.config.get('fast_transformer', False),
            pre_norm=False,
        )

        # Load pre-trained weights
        model_file = model_dir / "best_model.pt"
        print(f"Loading pre-trained scGPT weights from {model_file}")

        state_dict = torch.load(model_file, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        missing_keys, unexpected_keys = self.scgpt_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

        # Freeze scGPT
        if freeze_scgpt:
            for param in self.scgpt_model.parameters():
                param.requires_grad = False
            self.scgpt_model.eval()
            print("scGPT model frozen for inference")

    def forward(self, gene_ids: torch.Tensor, expression_values: torch.Tensor):
        """
        Get dynamic embeddings for genes based on their expression values

        Args:
            gene_ids: Gene token IDs, shape (n_genes,) - vocab indices for the genes
            expression_values: Gene expression values, shape (batch, n_genes)

        Returns:
            cell_emb: Cell-level embeddings, shape (batch, d_model=512)
        """
        device = expression_values.device
        batch_size = expression_values.shape[0]

        # Convert to numpy for tokenization
        if isinstance(gene_ids, torch.Tensor):
            gene_ids_np = gene_ids.cpu().numpy()
        else:
            gene_ids_np = np.array(gene_ids)

        expression_values_np = expression_values.cpu().numpy()

        # Tokenize and pad batch (filters non-zero genes, adds <cls> token)
        tokenized = tokenize_and_pad_batch(
            data=expression_values_np,
            gene_ids=gene_ids_np,
            max_len=self.max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=True,
            include_zero_gene=False,  # Only non-zero genes
            cls_token="<cls>",
            return_pt=True,
        )

        # Move to device
        src = tokenized['genes'].to(device)  # (batch, seq_len)
        values = tokenized['values'].to(device)  # (batch, seq_len)

        # Create padding mask
        src_key_padding_mask = (src == self.vocab[self.pad_token])

        # Forward through scGPT (frozen, no gradients)
        with torch.no_grad():
            output = self.scgpt_model(
                src=src,
                values=values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )

        # Extract cell embedding from <cls> token
        cell_emb = output['cell_emb']  # (batch, 512)

        return cell_emb


def create_gene_id_mapping(sc_adata, st_adata, vocab: GeneVocab):
    """
    Map AnnData gene names to vocab indices for both SC and ST data.
    Handles transposed data and mouse-to-human gene name conversion.

    Args:
        sc_adata: Single-cell AnnData object
        st_adata: Spatial transcriptomics AnnData object
        vocab: scGPT GeneVocab

    Returns:
        sc_gene_ids: numpy array of vocab indices for SC data, shape (n_sc_genes,)
        st_gene_ids: numpy array of vocab indices for ST data, shape (n_st_genes,)
    """
    pad_idx = vocab[vocab.pad_token] if hasattr(vocab, 'pad_token') and vocab.pad_token is not None else vocab['<pad>']

    def get_gene_names(adata):
        """Extract gene names, handling transposed data"""
        # Check if data appears transposed (var_names look like barcodes)
        var_sample = str(adata.var_names[0]) if len(adata.var_names) > 0 else ""
        obs_sample = str(adata.obs_names[0]) if len(adata.obs_names) > 0 else ""

        # Cell barcodes typically have dashes and look like "AAACCTGA-1"
        var_looks_like_barcode = '-' in var_sample and any(c.isdigit() for c in var_sample)
        obs_looks_like_gene = not ('-' in obs_sample and any(c.isdigit() for c in obs_sample))

        if var_looks_like_barcode and obs_looks_like_gene:
            print(f"  Data appears transposed! Using obs_names as genes")
            return adata.obs_names.tolist()
        else:
            return adata.var_names.tolist()

    def convert_mouse_to_human(gene_name):
        """Convert mouse gene names to human (simple uppercase conversion)"""
        if gene_name in vocab:
            return gene_name  # Already matches

        # Try uppercase (most human genes are uppercase)
        upper_name = gene_name.upper()
        if upper_name in vocab:
            return upper_name

        # Try capitalizing first letter only
        cap_name = gene_name[0].upper() + gene_name[1:] if len(gene_name) > 1 else gene_name.upper()
        if cap_name in vocab:
            return cap_name

        return None  # No match found

    def map_genes(adata, label):
        """Map genes for a dataset"""
        gene_names = get_gene_names(adata)
        gene_ids = []
        direct_matches = 0
        converted_matches = 0

        for g in gene_names:
            if g in vocab:
                gene_ids.append(vocab[g])
                direct_matches += 1
            else:
                # Try mouse-to-human conversion
                converted = convert_mouse_to_human(g)
                if converted and converted in vocab:
                    gene_ids.append(vocab[converted])
                    converted_matches += 1
                else:
                    gene_ids.append(pad_idx)

        gene_ids = np.array(gene_ids)

        # Report coverage
        total_matches = direct_matches + converted_matches
        coverage = total_matches / len(gene_ids) * 100
        print(f"{label} gene coverage: {coverage:.2f}% ({total_matches}/{len(gene_ids)} genes in vocab)")
        print(f"  Direct matches: {direct_matches}, Converted matches: {converted_matches}")

        return gene_ids

    sc_gene_ids = map_genes(sc_adata, "SC")
    st_gene_ids = map_genes(st_adata, "ST")

    return sc_gene_ids, st_gene_ids
