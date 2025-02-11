# dataset/copy_dataset.py
import torch
from torch.utils.data import Dataset
import random

class CopyDataset(Dataset):


    def __init__(self, num_samples, seq_len, vocab_size,
                 pad_token=0, sos_token=1, eos_token=2):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.data = []
        for _ in range(num_samples):

            seq = [random.randint(3, vocab_size - 1) for _ in range(seq_len - 2)]
            src_seq = [sos_token] + seq + [eos_token]
            tgt_seq = [sos_token] + seq + [eos_token]
            self.data.append((torch.tensor(src_seq, dtype=torch.long),
                              torch.tensor(tgt_seq, dtype=torch.long)))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
