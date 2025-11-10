#!/usr/bin/env python3
"""Script to update DiT_diff class with scGPT integration"""

# Read the original file
with open('model/diff_model_original.py', 'r') as f:
    content = f.read()

# Find and replace the __init__ method
init_old = """    def __init__(self,
                 st_input_size,
                 condi_input_size,
                 hidden_size,
                 depth,
                 dit_type,
                 num_heads,
                 classes,
                 pca_dim,
                 mlp_ratio=4.0,
                 **kwargs) -> None:
        super().__init__()

        self.st_input_size = st_input_size
        self.condi_input_size = condi_input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.classes = classes
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type
        self.pca_dim = pca_dim
        self.in_layer = nn.Sequential(
            nn.Linear(st_input_size, hidden_size),
            # nn.Dropout(p=0.5)
        )
        self.x_in_layer = nn.Sequential(
            nn.Linear(condi_input_size, hidden_size)
        )
        self.cond_layer = nn.Sequential(
            nn.Linear(self.condi_input_size, hidden_size),
            # nn.Dropout(p=0.5)
        )

        self.cond_layer_atten= SelfAttention2(self.condi_input_size, self.hidden_size)
        self.cond_layer_mlp = SimpleMLP(self.condi_input_size, self.hidden_size, self.hidden_size*2)
        # celltype emb
        self.condi_emb = nn.Embedding(classes, hidden_size)
        self.unet = UNet(in_features=hidden_size * 2, out_features=self.st_input_size)
        # time emb
        self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size *2)"""

init_new = """    def __init__(self,
                 st_input_size,
                 condi_input_size,
                 hidden_size,
                 depth,
                 dit_type,
                 num_heads,
                 classes,
                 pca_dim,
                 mlp_ratio=4.0,
                 use_scgpt=False,
                 scgpt_model_dir=None,
                 freeze_scgpt=True,
                 **kwargs) -> None:
        super().__init__()

        self.st_input_size = st_input_size
        self.condi_input_size = condi_input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.classes = classes
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type
        self.pca_dim = pca_dim
        self.use_scgpt = use_scgpt

        self.in_layer = nn.Sequential(
            nn.Linear(st_input_size, hidden_size),
            # nn.Dropout(p=0.5)
        )
        self.x_in_layer = nn.Sequential(
            nn.Linear(condi_input_size, hidden_size)
        )
        self.cond_layer = nn.Sequential(
            nn.Linear(self.condi_input_size, hidden_size),
            # nn.Dropout(p=0.5)
        )

        self.cond_layer_atten= SelfAttention2(self.condi_input_size, self.hidden_size)
        self.cond_layer_mlp = SimpleMLP(self.condi_input_size, self.hidden_size, self.hidden_size*2)

        # Condition embedding: use scGPT or simple embedding
        if use_scgpt and scgpt_model_dir is not None:
            print(f"Using scGPT embedder from {scgpt_model_dir}")
            self.scgpt_embedder = scGPTConditionEmbedder(
                model_dir=scgpt_model_dir,
                hidden_size=hidden_size * 2,  # Match time_emb size
                freeze_scgpt=freeze_scgpt,
                max_seq_len=1200,
                device=kwargs.get('device', 'cuda')
            )
            self.condi_emb = None  # Not used when scGPT is enabled
        else:
            print("Using standard embedding for condition")
            self.condi_emb = nn.Embedding(classes, hidden_size)
            self.scgpt_embedder = None

        self.unet = UNet(in_features=hidden_size * 2, out_features=self.st_input_size)
        # time emb
        self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size *2)"""

# Replace init
content = content.replace(init_old, init_new)

# Find and replace the forward method
forward_old = """    def forward(self, x, x_hat, t, y, **kwargs):
        x = x.float()
        x_hat = x_hat.float()
        x_hat = self.x_in_layer(x_hat)
        # x_hat = pca_with_torch(x_hat, self.pca_dim)
        t = self.time_emb(t)
        y = self.cond_layer_mlp(y)
        # y = self.cond_layer(y)
        # y = self.cond_layer_atten(y)
        # z = self.condi_emb(z)
        c = t + y
        # c = t

        x = self.in_layer(x)
        x = torch.cat([x, x_hat], dim=1)
        # for blk in self.blks:
        #     x = blk(x, c)
        # return self.out_layer(x, c)
        x = self.unet(x)
        return x"""

forward_new = """    def forward(self, x, x_hat, t, y, z=None, gene_ids=None, **kwargs):
        \"\"\"
        Args:
            x: noisy spatial transcriptomics data [batch_size, st_genes]
            x_hat: conditional single-cell data [batch_size, sc_genes]
            t: timestep [batch_size]
            y: single-cell expression for scGPT (if use_scgpt=True) [batch_size, sc_genes]
               or processed features for MLP (if use_scgpt=False)
            z: cell type labels [batch_size] (only used if use_scgpt=False)
            gene_ids: gene indices for scGPT tokenization [sc_genes] (optional)
        \"\"\"
        x = x.float()
        x_hat = x_hat.float()
        x_hat = self.x_in_layer(x_hat)
        # x_hat = pca_with_torch(x_hat, self.pca_dim)

        # Time embedding
        t = self.time_emb(t)

        # Condition embedding: use scGPT or simple methods
        if self.use_scgpt and self.scgpt_embedder is not None:
            # Use scGPT to embed single-cell data
            y = self.scgpt_embedder(sc_data=y, gene_ids=gene_ids)
        else:
            # Use original MLP-based conditioning
            y = self.cond_layer_mlp(y)
            # Alternative options (commented out in original):
            # y = self.cond_layer(y)
            # y = self.cond_layer_atten(y)
            # if z is not None and self.condi_emb is not None:
            #     z_emb = self.condi_emb(z)
            #     y = y + z_emb

        c = t + y
        # c = t

        x = self.in_layer(x)
        x = torch.cat([x, x_hat], dim=1)
        # for blk in self.blks:
        #     x = blk(x, c)
        # return self.out_layer(x, c)
        x = self.unet(x)
        return x"""

# Replace forward
content = content.replace(forward_old, forward_new)

# Add import at the top
import_line = "from preprocess.utils import pca_with_torch\n"
new_import = "from preprocess.utils import pca_with_torch\nfrom model.scgpt_embedder import scGPTConditionEmbedder, scGPTConditionEmbedderSimple\n"
content = content.replace(import_line, new_import)

# Write the updated file
with open('model/diff_model.py', 'w') as f:
    f.write(content)

print("Successfully updated diff_model.py with scGPT integration!")
