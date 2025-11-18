"""
Check the actual structure of the dataset to understand alignment
"""
import scanpy as sc
import torch

sc_path = 'datasets/dataset_ML/sc/dataset_ML_sc.h5ad'
st_path = 'datasets/dataset_ML/st/dataset_ML_st.h5ad'

print("Loading data...")
sc_data = sc.read_h5ad(sc_path)
st_data = sc.read_h5ad(st_path)

print("\n=== Original h5ad structure ===")
print(f"SC data shape: {sc_data.shape} (n_obs x n_vars)")
print(f"  obs (rows): {len(sc_data.obs)} - probably cells/spots")
print(f"  var (cols): {len(sc_data.var)} - probably genes")
print(f"  First obs name: {sc_data.obs_names[0]}")
print(f"  First var name: {sc_data.var_names[0]}")

print(f"\nST data shape: {st_data.shape} (n_obs x n_vars)")
print(f"  obs (rows): {len(st_data.obs)} - probably cells/spots")
print(f"  var (cols): {len(st_data.var)} - probably genes")
print(f"  First obs name: {st_data.obs_names[0]}")
print(f"  First var name: {st_data.var_names[0]}")

print("\n=== After transpose (as done in dataset) ===")
st_df = st_data.to_df().T
sc_df = sc_data.to_df().T

print(f"SC transposed shape: {sc_df.shape} (genes x cells)")
print(f"  Rows (genes): {len(sc_df)}")
print(f"  Cols (cells): {len(sc_df.columns)}")

print(f"\nST transposed shape: {st_df.shape} (genes x spots)")
print(f"  Rows (genes): {len(st_df)}")
print(f"  Cols (spots): {len(st_df.columns)}")

print("\n=== After converting to tensor ===")
st_tensor = torch.tensor(st_df.values, dtype=torch.float32)
sc_tensor = torch.tensor(sc_df.values, dtype=torch.float32)

print(f"SC tensor shape: {sc_tensor.shape}")
print(f"ST tensor shape: {st_tensor.shape}")

print("\n=== When you do st_sample[idx] ===")
print(f"st_sample[0] would have shape: {st_tensor[0].shape}")
print(f"sc_sample[0] would have shape: {sc_tensor[0].shape}")

print("\n=== The issue ===")
print(f"Dataset __len__ returns: {len(st_df)} (number of genes)")
print(f"So idx ranges from 0 to {len(st_df)-1}")
print(f"This means you're iterating over GENES, not spots/cells!")
print(f"st_sample[idx] gives expression of gene idx across ALL {st_tensor.shape[1]} spots")
print(f"sc_sample[idx] gives expression of gene idx across ALL {sc_tensor.shape[1]} cells")
