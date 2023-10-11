import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        # linear layers for key, query and value
        self.key = nn.Linear(number_embed, head_size, bias=False)
        self.query = nn.Linear(number_embed, head_size, bias=False)
        self.value = nn.Linear(number_embed, head_size, bias=False)
        # buffer for the lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-1, -2) / (k.shape[-1] ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class multiHead(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        # multiple heads 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # linear layer to project the concatenated heads
        self.proj = nn.Linear(number_embed, number_embed)
        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class feedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # MLP layers
        self.net = nn.Sequential(
            # first linear layer with ReLU activation
            nn.Linear(number_embed,  number_embed * inner_linear_factor),
            nn.ReLU(),
            # second linear layer for residual connection
            nn.Linear(number_embed * inner_linear_factor,  number_embed),
            # dropout layer
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        head_size = number_embed // number_head
        # multi-head attention
        self.sa = multiHead(number_head, head_size)
        # feed forward layer
        self.ff = feedForward()
        # layer norms
        self.ln1 = nn.LayerNorm(number_embed)
        self.ln2 = nn.LayerNorm(number_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x