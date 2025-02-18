import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, src):
        # src: (B, N, embed_dim)
        src2 = self.norm1(src)
        attn_output, _ = self.attn(src2, src2, src2)
        src = src + self.dropout1(attn_output)
        src2 = self.norm2(src)
        src = src + self.mlp(src2)
        return src

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, embed_dim, num_heads, mlp_ratio=4.0, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src):
        # src: (B, N, embed_dim)
        for layer in self.layers:
            src = layer(src)
        src = self.norm(src)
        return src
