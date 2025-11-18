import scipy
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from scipy.sparse import issparse, csr
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
CHUNK_SIZE = 20000

# class ConditionalDiffusionDataset(Dataset):
#     def __init__(self, sc_path, st_path, transform=None):
#         # 读取数据
#         self.sc_data = sc.read_h5ad(sc_path)
#         self.st_data = sc.read_h5ad(st_path)
#         self.transform = transform
#
#         self.max_len = max(len(self.sc_data), len(self.st_data))
#         sc_df = self.sc_data.to_df()
#         st_df = self.st_data.to_df()
#
#         if len(sc_df) < self.max_len:
#             sc_df = sc_df.sample(n=self.max_len, replace=True)
#         if len(st_df) < self.max_len:
#             st_df = st_df.sample(n=self.max_len, replace=True)
#
#         sc_df = sc_df.iloc[:self.max_len]
#         st_df = st_df.iloc[:self.max_len]
#
#         self.sc_data = sc_df
#         self.st_data = st_df
#
#     def __len__(self):
#         return self.max_len
#
#     def __getitem__(self, idx):
#         sc_sample = self.sc_data.iloc[idx]
#         st_sample = self.st_data.iloc[idx]
#         if self.transform:
#             sc_sample, st_sample = self.transform(sc_sample, st_sample)
#         sc_sample = torch.tensor(sc_sample.values, dtype=torch.float32)
#         st_sample = torch.tensor(st_sample.values, dtype=torch.float32)
#
#         return st_sample, sc_sample

class ConditionalDiffusionDataset(Dataset):
    def __init__(self, sc_path, st_path, gene_ids=None):
        """
        Dataset for conditional diffusion model

        Args:
            sc_path: Path to scRNA-seq h5ad file
            st_path: Path to spatial transcriptomics h5ad file
            gene_ids: Optional numpy array of gene vocabulary indices for scGPT, shape (n_sc_genes,)
        """
        self.sc_data = sc.read_h5ad(sc_path)
        self.st_data = sc.read_h5ad(st_path)

        # Convert to dataframes (keep original orientation: rows=spots/cells, cols=genes)
        self.st_data_df = self.st_data.to_df()  # (n_st_spots, n_st_genes)
        self.sc_data_df = self.sc_data.to_df()  # (n_sc_cells, n_sc_genes)

        self.gene_names = self.st_data.var_names.tolist()  # ST gene names
        self.sc_gene_names = self.sc_data.var_names.tolist()  # SC gene names

        # Convert to tensors: (n_spots/cells, n_genes)
        self.st_sample = torch.tensor(self.st_data_df.values, dtype=torch.float32)  # (n_st_spots, n_st_genes)
        self.sc_sample = torch.tensor(self.sc_data_df.values, dtype=torch.float32)  # (n_sc_cells, n_sc_genes)

        # Store gene IDs for scGPT
        self.gene_ids = gene_ids
        if gene_ids is not None:
            self.gene_ids = torch.tensor(gene_ids, dtype=torch.long)

    def __len__(self):
        # Return number of ST spots (we iterate over spatial spots)
        return len(self.st_sample)

    def __getitem__(self, idx):
        # Get ST spot at idx
        st_spot = self.st_sample[idx]  # (n_st_genes,)

        # For SC conditioning, we need to handle the mismatch between n_st_spots (2177) and n_sc_cells (887)
        # Strategy: cycle through SC cells (when idx >= n_sc_cells, wrap around)
        sc_idx = idx % len(self.sc_sample)
        sc_cell = self.sc_sample[sc_idx]  # (n_sc_genes,)

        # Return: ST spot, SC cell (for x_hat), SC cell (for scGPT conditioning), gene_ids
        if self.gene_ids is not None:
            return st_spot, sc_cell, sc_cell, self.gene_ids
        else:
            return st_spot, sc_cell, sc_cell

    def get_gene_names(self):
        return self.gene_names

    def get_sc_gene_names(self):
        return self.sc_gene_names



def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list

    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks

    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        for i in range(new_X.shape[0] // chunk_size + 1):
            new_X[i * chunk_size:(i + 1) * chunk_size, idx] = adata[i * chunk_size:(i + 1) * chunk_size, genes[idx]].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={'var_names': genes})
    return adata


def plot_hvg_umap(hvg_adata, color=['celltype'], save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3, 3))  # type: ignore
    hvg_adata = hvg_adata.copy()
    if save_filename:
        sc.settings.figdir = save_filename
        save = '.pdf'
    else:
        save = None
    # ideal gas equation

    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata)
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(hvg_adata, min_dist=0.1)
    sc.pl.umap(hvg_adata, color=color, legend_fontsize=10, ncols=2, show=None, save=save, wspace=1)
    return hvg_adata


def get_data_loader(data_ary: np.ndarray,
                    cell_type: np.ndarray,
                    batch_size: int = 512,
                    is_shuffle: bool = True,
                    ):
    data_tensor = torch.from_numpy(data_ary.astype(np.float32))
    cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))
    dataset = TensorDataset(data_tensor, cell_type_tensor)
    generator = torch.Generator(device='cuda')
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False,
        generator=generator)  # , generator=torch.Generator(device = 'cuda')


def scale(adata):
    scaler = MaxAbsScaler()
    # 对adata.X按行进行归一化
    normalized_data = scaler.fit_transform(adata.X.T).T

    # 更新归一化后的数据到adata.X
    adata.X = normalized_data
    return adata


def data_augment(adata: AnnData, fixed: bool, noise_std):
    # 定义增强参数，例如噪声的标准差
    noise_stddev = noise_std
    augmented_adata = adata.copy()
    gene_expression = adata.X

    if fixed:
        augmented_adata.X = augmented_adata.X + np.full(gene_expression.shape, noise_stddev)
    else:
        # 对每个基因的表达值引入随机噪声
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_stddev, gene_expression.shape))

    merge_adata = adata.concatenate(augmented_adata, join='outer')

    return merge_adata




