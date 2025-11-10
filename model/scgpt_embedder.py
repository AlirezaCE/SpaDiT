"""
scGPT-based condition embedder for DiT diffusion model.
This module loads a pretrained scGPT model and uses it to generate
rich biological embeddings from single-cell gene expression data.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import sys

# Add scGPT to path
sys.path.append("../scGPT")

from scGPT.scgpt.model import TransformerModel
from scGPT.scgpt.tokenizer.gene_tokenizer import GeneVocab
from scGPT.scgpt.tokenizer import tokenize_and_pad_batch


class scGPTConditionEmbedder(nn.Module):
    """
    Wrapper for scGPT model to generate condition embeddings from single-cell data.

    Args:
        model_dir: Path to the pretrained scGPT model directory
        hidden_size: Target hidden size for the diffusion model
        freeze_scgpt: Whether to freeze scGPT parameters (recommended for stability)
        max_seq_len: Maximum sequence length for tokenization
        pad_token: Padding token name
        use_batch_labels: Whether scGPT uses batch labels
    """

    def __init__(
        self,
        model_dir: str,
        hidden_size: int,
        freeze_scgpt: bool = True,
        max_seq_len: int = 1200,
        pad_token: str = "<pad>",
        use_batch_labels: bool = False,
        device: str = "cuda"
    ):
        super().__init__()

        self.model_dir = Path(model_dir)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        self.pad_value = 0
        self.use_batch_labels = use_batch_labels
        self.device = device

        # Load vocabulary
        vocab_file = self.model_dir / "vocab.json"
        self.vocab = GeneVocab.from_file(vocab_file)

        # Add special tokens if not present
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab.append_token(token)

        # Load model config
        config_file = self.model_dir / "args.json"
        with open(config_file, "r") as f:
            self.model_config = json.load(f)

        # Create scGPT model
        self.scgpt_model = self._create_scgpt_model()

        # Load pretrained weights
        model_file = self.model_dir / "best_model.pt"
        self._load_pretrained_weights(model_file)

        # Move model to device BEFORE freezing
        self.scgpt_model.to(self.device)

        # Freeze scGPT if requested
        if freeze_scgpt:
            for param in self.scgpt_model.parameters():
                param.requires_grad = False
            print("scGPT parameters frozen")

        # Projection layer to map scGPT embedding to target hidden_size
        scgpt_dim = self.model_config.get("embsize", 512)
        self.projection = nn.Sequential(
            nn.Linear(scgpt_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        print(f"scGPT embedder initialized: {scgpt_dim} -> {hidden_size}")

    def _create_scgpt_model(self):
        """Create scGPT model from config"""
        model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.model_config["embsize"],
            nhead=self.model_config["nheads"],
            d_hid=self.model_config["d_hid"],
            nlayers=self.model_config["nlayers"],
            nlayers_cls=self.model_config.get("n_layers_cls", 3),
            vocab=self.vocab,
            dropout=self.model_config.get("dropout", 0.2),
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=self.model_config.get("do_mvc", False),
            do_dab=self.model_config.get("do_dab", False),
            use_batch_labels=self.use_batch_labels,
            domain_spec_batchnorm=self.model_config.get("domain_spec_batchnorm", False),
            input_emb_style=self.model_config.get("input_emb_style", "continuous"),
            cell_emb_style=self.model_config.get("cell_emb_style", "cls"),
            explicit_zero_prob=self.model_config.get("explicit_zero_prob", False),
            use_fast_transformer=self.model_config.get("use_fast_transformer", False),
            pre_norm=self.model_config.get("pre_norm", False),
        )
        return model

    def _load_pretrained_weights(self, model_file: Path):
        """Load pretrained weights with proper handling"""
        try:
            state_dict = torch.load(model_file, map_location='cpu')  # Load to CPU first

            # Handle different checkpoint formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            # Load with strict=False to handle missing keys
            missing_keys, unexpected_keys = self.scgpt_model.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                print(f"Missing keys in checkpoint: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")

            print(f"Loaded pretrained scGPT weights from {model_file}")

        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with randomly initialized scGPT model")

    def tokenize_sc_data(self, sc_data: torch.Tensor, gene_ids: torch.Tensor):
        """
        Tokenize single-cell data for scGPT input.

        Args:
            sc_data: [batch_size, n_genes] expression values
            gene_ids: [n_genes] gene indices in vocabulary

        Returns:
            Dictionary with tokenized data
        """
        batch_size = sc_data.shape[0]

        # Convert to numpy for tokenization
        sc_data_np = sc_data.cpu().numpy()
        gene_ids_np = gene_ids.cpu().numpy()

        # Tokenize and pad
        tokenized = tokenize_and_pad_batch(
            data=sc_data_np,
            gene_ids=gene_ids_np,
            max_len=self.max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=True,
            include_zero_gene=False,
            cls_token="<cls>",
            return_pt=True,
        )

        return tokenized

    def forward(self, sc_data: torch.Tensor, gene_ids: torch.Tensor = None,
                batch_labels: torch.Tensor = None):
        """
        Generate embeddings from single-cell data using scGPT.

        Args:
            sc_data: [batch_size, n_genes] single-cell gene expression values
            gene_ids: [n_genes] gene indices in vocabulary (optional, can be precomputed)
            batch_labels: [batch_size] batch labels if use_batch_labels=True

        Returns:
            embeddings: [batch_size, hidden_size] condition embeddings
        """
        batch_size = sc_data.shape[0]

        # If gene_ids not provided, use all genes in vocab
        if gene_ids is None:
            gene_ids = torch.arange(sc_data.shape[1], device=sc_data.device)

        # Tokenize the data
        tokenized = self.tokenize_sc_data(sc_data, gene_ids)

        # Move to device
        input_gene_ids = tokenized["genes"].to(self.device)
        input_values = tokenized["values"].to(self.device)

        # Create padding mask
        src_key_padding_mask = (input_gene_ids == self.vocab[self.pad_token])

        # Encode with scGPT
        with torch.set_grad_enabled(self.training):
            # Get cell embeddings from scGPT
            output = self.scgpt_model._encode(
                src=input_gene_ids,
                values=input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if self.use_batch_labels else None
            )

            # Extract cell embeddings (using the method from scGPT)
            cell_embeddings = self.scgpt_model._get_cell_emb_from_layer(
                output, weights=input_values
            )

        # Project to target hidden size
        embeddings = self.projection(cell_embeddings)

        return embeddings


class scGPTConditionEmbedderSimple(nn.Module):
    """
    Simplified version that processes already-embedded sc data.
    Use this if you want to precompute scGPT embeddings to save time.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, sc_embeddings: torch.Tensor):
        """
        Args:
            sc_embeddings: [batch_size, input_dim] precomputed scGPT embeddings
        Returns:
            [batch_size, hidden_size] projected embeddings
        """
        return self.projection(sc_embeddings)
