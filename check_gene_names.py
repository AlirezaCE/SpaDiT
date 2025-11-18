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

    print(f"AnnData shape: {adata.shape} (n_obs x n_vars)")
    print(f"adata.var_names (should be genes): {len(adata.var_names)} entries")
    print(f"adata.obs_names (should be cells/spots): {len(adata.obs_names)} entries")

    data_genes = adata.var_names.tolist()
    data_obs = adata.obs_names.tolist()

    print(f"\nFirst 20 var_names: {data_genes[:20]}")
    print(f"First 20 obs_names: {data_obs[:20]}")

    # Check which one looks like genes
    print("\n=== Checking var_names ===")
    data_genes = adata.var_names.tolist()
    print(f"Dataset has {len(data_genes)} var_names")

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

    # Now check obs_names (in case axes are swapped)
    print("\n=== Checking obs_names (in case data is transposed) ===")
    data_genes_from_obs = adata.obs_names.tolist()
    print(f"Dataset has {len(data_genes_from_obs)} obs_names")

    data_genes_from_obs_set = set(data_genes_from_obs)
    overlap_obs = vocab_genes & data_genes_from_obs_set
    print(f"Gene overlap (using obs_names): {len(overlap_obs)} / {len(data_genes_from_obs)} ({len(overlap_obs)/len(data_genes_from_obs)*100:.2f}%)")

    if overlap_obs:
        print(f"Example matching genes: {list(overlap_obs)[:10]}")

    if len(overlap_obs) > len(overlap):
        print("\n⚠️  WARNING: obs_names has better gene overlap than var_names!")
        print("     Your h5ad file appears to be TRANSPOSED (rows/columns swapped)")
        print("     You may need to transpose with: adata = adata.T")

    # Test mouse-to-human gene conversion
    print("\n=== Testing Mouse-to-Human Gene Conversion ===")

    def convert_mouse_to_human(gene_name, vocab_set):
        """Simple mouse to human gene name conversion"""
        if gene_name in vocab_set:
            return gene_name  # Already matches

        # Try uppercase (most human genes are uppercase)
        upper_name = gene_name.upper()
        if upper_name in vocab_set:
            return upper_name

        # Try capitalizing first letter only (some genes like Xkr4 -> XKR4)
        cap_name = gene_name[0].upper() + gene_name[1:] if len(gene_name) > 1 else gene_name.upper()
        if cap_name in vocab_set:
            return cap_name

        return None  # No match found

    # Use obs_names since they contain the actual genes
    mouse_genes = data_genes_from_obs
    converted_matches = 0
    examples = []

    for gene in mouse_genes[:100]:  # Test first 100 genes
        converted = convert_mouse_to_human(gene, vocab_genes)
        if converted and converted != gene:
            converted_matches += 1
            if len(examples) < 10:
                examples.append(f"{gene} -> {converted}")

    print(f"Tested first 100 genes:")
    print(f"  Direct matches: {len([g for g in mouse_genes[:100] if g in vocab_genes])}")
    print(f"  Conversion matches: {converted_matches}")
    print(f"  Total matches: {len([g for g in mouse_genes[:100] if g in vocab_genes]) + converted_matches}")

    if examples:
        print(f"\nExample conversions:")
        for ex in examples:
            print(f"  {ex}")

    # Now test on ALL genes
    print("\n=== Testing on ALL genes ===")
    all_matches = 0
    direct_matches = 0
    converted_matches_all = 0

    for gene in mouse_genes:
        if gene in vocab_genes:
            direct_matches += 1
            all_matches += 1
        else:
            converted = convert_mouse_to_human(gene, vocab_genes)
            if converted:
                converted_matches_all += 1
                all_matches += 1

    print(f"Total genes: {len(mouse_genes)}")
    print(f"Direct matches: {direct_matches} ({direct_matches/len(mouse_genes)*100:.2f}%)")
    print(f"Conversion matches: {converted_matches_all} ({converted_matches_all/len(mouse_genes)*100:.2f}%)")
    print(f"Total coverage: {all_matches} / {len(mouse_genes)} ({all_matches/len(mouse_genes)*100:.2f}%)")
