"""
Main script for SpaDiT with scGPT embeddings
This version uses scGPT pre-trained embeddings for conditioning
"""

import anndata as ad
import numpy as np
import pandas as pd
import sys
import pickle
import os
import datetime
import time as tm
from functools import partial
import scipy.stats as st
from scipy.stats import wasserstein_distance
import scipy.stats
import copy
from sklearn.model_selection import KFold
import pandas as pd
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import seaborn as sns
import torch
from scipy.spatial.distance import cdist
import h5py
import time
import sys
import tangram as tg
import pickle
import yaml
import argparse
from os.path import join
from IPython.display import display

from model.diff_model import DiT_diff
from model.diff_scheduler import NoiseScheduler
from model.diff_train import normal_train_diff
from model.sample import sample_diff
from model.scgpt_wrapper import create_gene_id_mapping
from preprocess.result_analysis import clustering_metrics
from preprocess.utils import *
from preprocess.data import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SpaDiT with scGPT embeddings')
parser.add_argument("--sc_data", type=str, default='_sc.h5ad')
parser.add_argument("--st_data", type=str, default='_st.h5ad')
parser.add_argument("--document", type=str, default='dataset_ML')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--diffusion_step", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--depth", type=int, default=16)
parser.add_argument("--noise_std", type=float, default=10)
parser.add_argument("--pca_dim", type=int, default=100)
parser.add_argument("--head", type=int, default=16)
parser.add_argument("--mask_nonzero_ratio", type=float, default=0.3)
parser.add_argument("--mask_zero_ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=3407)

# scGPT parameters
parser.add_argument("--use_scgpt", action='store_true', help='Use scGPT embeddings for conditioning')
parser.add_argument("--scgpt_model_dir", type=str, default='scgpt_models/scGPT_brain',
                    help='Path to scGPT pre-trained model')

args = parser.parse_args()

print(os.getcwd())
print(f"Using device: {args.device}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


def train_valid_test():
    seed_everything(args.seed)
    st_path = 'datasets/' + args.document + '/st/' + args.document + args.st_data
    sc_path = 'datasets/' + args.document + '/sc/' + args.document + args.sc_data

    directory = 'save/' + args.document + '_ckpt/' + args.document + '_scdiff'

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_suffix = '_scgpt' if args.use_scgpt else ''
    save_path = os.path.join(directory, args.document + model_suffix + '.pt')

    # Load data for gene ID mapping if using scGPT
    gene_ids = None
    if args.use_scgpt:
        print("\n=== Loading scGPT vocabulary and creating gene mappings ===")
        from scgpt.tokenizer import GeneVocab

        # Load scGPT vocabulary
        vocab_path = os.path.join(args.scgpt_model_dir, 'vocab.json')
        vocab = GeneVocab.from_file(vocab_path)
        print(f"Loaded scGPT vocabulary with {len(vocab)} genes")

        # Load data to get gene names
        sc_adata = sc.read_h5ad(sc_path)
        st_adata = sc.read_h5ad(st_path)

        # Create gene ID mappings
        sc_gene_ids, st_gene_ids = create_gene_id_mapping(sc_adata, st_adata, vocab)

        # Use SC gene IDs for conditioning
        gene_ids = sc_gene_ids
        print(f"Gene IDs created: {len(gene_ids)} genes mapped to vocab")

    # Create dataset with gene IDs
    dataset = ConditionalDiffusionDataset(sc_path, st_path, gene_ids=gene_ids)

    (train_dataset, train_gene_names), (valid_dataset, valid_gene_names), (
        test_dataset, test_gene_names) = split_dataset_with_gene_names(dataset, train_ratio=0.7, val_ratio=0.2,
                                                                       test_ratio=0.1, random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    cell_num = dataset.sc_data.shape[1]
    spot_num = dataset.st_data.shape[1]
    sc_gene_num = dataset.sc_data.shape[0]
    st_gene_num = dataset.st_data.shape[0]

    print(f"\nDataset info:")
    print(f"  SC genes: {sc_gene_num}, spots/cells: {cell_num}")
    print(f"  ST genes: {st_gene_num}, spots: {spot_num}")
    print(f"  Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    # Initialize model with scGPT
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
        use_scgpt=args.use_scgpt,
        scgpt_model_dir=args.scgpt_model_dir if args.use_scgpt else None,
    )

    model.to(args.device)
    diffusion_step = args.diffusion_step

    # Training
    if not os.path.isfile(save_path):
        print(f"\n=== Training model {'with scGPT' if args.use_scgpt else 'without scGPT'} ===")
        model.train()
        normal_train_diff(model,
                          dataloader=train_dataloader,
                          lr=args.learning_rate,
                          num_epoch=args.epoch,
                          diffusion_step=diffusion_step,
                          device=args.device,
                          pred_type='noise',
                          mask_nonzero_ratio=args.mask_nonzero_ratio,
                          mask_zero_ratio=args.mask_zero_ratio)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    else:
        print(f"Loading model from {save_path}")
        model.load_state_dict(torch.load(save_path))

    # Inference
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine',
        device=args.device
    )

    model.eval()

    with torch.no_grad():
        test_gt = torch.stack([data[0] for data in test_dataset])  # ST data
        test_sc = torch.stack([data[1] for data in test_dataset])  # SC data

        print(f"\n=== Running inference ===")
        prediction = sample_diff(model,
                                device=args.device,
                                dataloader=test_dataloader,
                                noise_scheduler=noise_scheduler,
                                mask_nonzero_ratio=0.3,
                                mask_zero_ratio=0,
                                gt=test_gt,
                                sc=test_sc,
                                num_step=diffusion_step,
                                sample_shape=(test_gt.shape[0], test_gt.shape[1]),
                                is_condi=True,
                                sample_intermediate=diffusion_step,
                                model_pred_type='x_start',
                                is_classifier_guidance=False,
                                omega=0.9)

    return prediction, test_gt, test_gene_names


# Main execution
Data = args.document
outdir = 'result/' + Data + '/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Save hyperparameters
hyper_directory = 'save/' + Data + '_ckpt/' + Data + '_hyper/'
hyper_file = Data + ('_scgpt_' if args.use_scgpt else '_') + 'hyperameters.yaml'
hyper_full_path = os.path.join(hyper_directory, hyper_file)
if not os.path.exists(hyper_directory):
    os.makedirs(hyper_directory)

args_dict = vars(args)
with open(hyper_full_path, 'w') as yaml_file:
    yaml.dump(args_dict, yaml_file)

print(f"\n{'='*80}")
print(f"SpaDiT {'WITH scGPT' if args.use_scgpt else 'WITHOUT scGPT'}")
print(f"{'='*80}\n")

# Run training and testing
prediction_result, ground_truth, test_gene_num = train_valid_test()

# Save results
gene_name = test_gene_num
prediction_result = prediction_result.T
ground_truth = ground_truth.numpy().T

pred_result = pd.DataFrame(prediction_result, columns=[gene_name])
original = pd.DataFrame(ground_truth, columns=[gene_name])

output_suffix = '_scgpt' if args.use_scgpt else ''
pred_result.to_csv(outdir + f'/SpaDiT{output_suffix}_prediction.csv', header=True, index=True)
original.to_csv(outdir + f'/original{output_suffix}.csv', header=True, index=True)

print(f"\n{'='*80}")
print(f"Results saved to {outdir}")
print(f"  - Predictions: SpaDiT{output_suffix}_prediction.csv")
print(f"  - Ground truth: original{output_suffix}.csv")
print(f"{'='*80}\n")
