import torch
import torch.nn as nn
import torch.nn.functional as F


def split_image_to_patches(x, patch_size):
    b, c, h, w = x.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x = x.reshape(b, c, num_patches_h, num_patches_w, patch_size, patch_size)
    x = x.permute(0, 2, 3, 1, 4, 5)
    
    return x.reshape(b, num_patches_h*num_patches_w, c, patch_size, patch_size)


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.attn_head = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attn_head(x, x, x)
        # Add & Norm
        x = self.ln1(x + attn_output)
        # MLP
        mlp_output = self.mlp(x)
        # Add & Norm
        x = self.ln2(x + mlp_output)
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_heads, num_layers):
        super().__init__()
        assert image_size % patch_size == 0, "image size must be divisible by patch size"
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.linear_emb = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(p=0.1)

        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == w == self.image_size, f"input image size must be {self.image_size}"

        # Extract patches
        x = split_image_to_patches(x, self.patch_size) # return shape (batch_size, num_patches, channels, patch_size, patch_size)
        x = x.reshape(b, self.num_patches, -1)
        x = self.linear_emb(x)

        # Add position embedding
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embedding[:, : x.size(1), :]

        # # Transformer blocks
        # for block in self.blocks:
        #     x = block(x)

        # # Pooling
        # x = x[:, 0]  # cls_token
        # x = self.dropout(x)

        # # MLP head
        # x = self.mlp_head(x)

        return x
