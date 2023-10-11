import torch
import tiktoken

torch.manual_seed(0)

def load_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    return text

def process_data(text):
    enc = tiktoken.get_encoding("cl100k_base")
    token_size = enc.n_vocab
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return enc, train_data, val_data, token_size

def get_batch(split, train_data, val_data, context_size, batch_size, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y