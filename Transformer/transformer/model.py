import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
import torch

def create_masks(src, tgt, pad_token=0):

    src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)
    tgt_seq_len = tgt.size(1)
    subsequent_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)).bool()
    tgt_mask = tgt_mask & subsequent_mask.unsqueeze(0).unsqueeze(0)
    return src_mask, tgt_mask

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_layers=6, num_heads=8, d_ff=2048,
                 dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        output = self.fc_out(output)
        return output
