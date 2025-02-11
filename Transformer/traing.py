import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer.model import Transformer, create_masks
from dataset.copy_dataset import CopyDataset

def train_transformer():
    # ハイパーパラメータの設定
    src_vocab_size = 50   
    tgt_vocab_size = 50   
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 512
    dropout = 0.1
    max_len = 20         
    pad_token = 0
    sos_token = 1
    eos_token = 2

    num_samples = 1000
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(src_vocab_size, tgt_vocab_size,
                        d_model, num_layers, num_heads, d_ff, dropout, max_len).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # CopyDataset の初期化
    dataset = CopyDataset(num_samples, max_len, src_vocab_size, pad_token, sos_token, eos_token)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for src, tgt in dataloader:
            src = src.to(device)  # (batch, seq_len)
            tgt = tgt.to(device)  # (batch, seq_len)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask, _ = create_masks(src, src, pad_token)
            _, tgt_mask = create_masks(tgt_input, tgt_input, pad_token)

            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            output = output.reshape(-1, tgt_vocab_size)
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_transformer()
