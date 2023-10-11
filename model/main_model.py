import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from model.model_components import *

class LanguageModel(nn.Module):

    def __init__(self, token_size):
        super().__init__()
        # word embedding
        self.token_embedding_table = nn.Embedding(token_size, number_embed)
        # position embedding
        self.position_embedding_table = nn.Embedding(context_size, number_embed)
        # multiple transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_block)])
        # final layer norm
        self.ln_final = nn.LayerNorm(number_embed)
        # linear layer to get logits
        self.lm_head = nn.Linear(number_embed, token_size)

    def forward(self, idx, targets=None):
        # idx is (B, T) array of indices in the current context
        # targets is (B, T) array of indices in the next context
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to context size
            idx_cond = idx[:, -context_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] 
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) 
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) 

            idx = torch.cat((idx, idx_next), dim=1) 
        return idx