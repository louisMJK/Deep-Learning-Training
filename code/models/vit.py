import torch
from torch import nn
import math

from ._builder import build_model_with_cfg
from ._registry import register_model


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_hiddens):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(in_channels, num_hiddens, kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        # input shape:  (batch_size, in_channels, img_size, img_size)
        # return shape: (batch_size, num_patches, num_hiddens)
        return self.conv(X).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_heads, dropout=0.1):
        super().__init__()
        assert num_hiddens % num_heads == 0
        self.num_heads = num_heads
        self.dropout = dropout
        qkv_bias = False
        self.W_q = nn.LazyLinear(num_hiddens, bias=qkv_bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=qkv_bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=qkv_bias)
        self.W_h = nn.Linear(num_hiddens, num_hiddens)

    def dot_product_attention(self, Q, K, V):
        # input shape:  (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        # output shape: (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        d = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d)  # (batch_size, num_heads, num_patches, num_patches)
        A = nn.Softmax(dim=-1)(scores)
        H = torch.matmul(nn.Dropout(self.dropout)(A), V)  # (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        return H
    
    def split_heads(self, X):
        # input:  (batch_size, num_patches, num_hiddens)
        # output: (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        return X.reshape(X.shape[0], X.shape[1], self.num_heads, -1).transpose(1, 2)
    
    def concat_heads(self, X):
        # input:  (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        # output: (batch_size, num_patches, num_hiddens)
        X = X.transpose(1,2)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, X):
        # input shape:  (batch_size, num_patches, in_hiddens)
        # return shape: (batch_size,)
        Q = self.split_heads(self.W_q(X))  # (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        H = self.dot_product_attention(Q, K, V)  # (batch_size, num_heads, num_patches, num_hiddens/num_heads)
        H = self.W_h(self.concat_heads(H))  # (batch_size, num_patches, num_hiddens)
        return H


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_heads, mlp_hiddens, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_hiddens)
        self.attention = MultiHeadSelfAttention(in_channels, num_hiddens, num_heads, dropout)
        self.norm2 = nn.LayerNorm(num_hiddens)
        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens, mlp_hiddens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hiddens, num_hiddens),
            nn.Dropout(dropout)
        )
    
    def forward(self, X):
        X = X + self.attention(self.norm1(X))
        X = X + self.mlp(self.norm2(X))
        return X


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_hiddens=384, num_heads=6, mlp_ratio=4, num_layers=6, num_classes=100, dropout=0.1):
        super().__init__()
        self.patch_embeding = PatchEmbedding(img_size, patch_size, in_channels, num_hiddens)
        self.class_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        self.pos_embedding = nn.Parameter(0.02 * torch.randn(1, self.patch_embeding.num_patches + 1, num_hiddens))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f"{i}", TransformerBlock(in_channels, num_hiddens, num_heads, mlp_ratio*num_hiddens, dropout))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, num_classes)
        )

    def forward(self, X):
        X = self.patch_embeding(X)
        X = torch.cat((self.class_token.expand(X.shape[0], -1, -1), X), dim=1)
        X = self.dropout(X + self.pos_embedding)
        for block in self.blocks:
            X = block(X)
        X = self.mlp_head(X[:, 0])
        return X


def _create_vit(variant, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model = build_model_with_cfg(VisionTransformer, variant, **kwargs)
    return model


@register_model
def vit_tiny(**kwargs):
    model_cfg = dict(
        num_hiddens=192, 
        num_heads=3, 
        num_layers=3, 
        **kwargs
    )
    model = _create_vit('vit_tiny', **model_cfg)
    return model

