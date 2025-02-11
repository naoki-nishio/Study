
import torch.nn as nn
from .layers import PositionalEncoding, MultiHeadAttention, FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask=None):
        src2 = self.self_attn(src, src, src, mask)
        src = self.norm1(src + self.dropout(src2))
        src2 = self.ff(src)
        src = self.norm2(src + self.dropout(src2))
        return src

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask=None):
        x = self.embedding(src) * (self.embedding.embedding_dim ** 0.5)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
