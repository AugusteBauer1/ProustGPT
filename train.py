import torch

from config import *
from data.prepare_data import process_data, load_data, get_batch
from model.main_model import LanguageModel
from model.utils import estimate_loss

text = load_data('data/ProustTempsPerdu.txt')

enc, train_data, val_data, token_size = process_data(text)

model = LanguageModel(token_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss(model, eval_iters, train_data, val_data, context_size, batch_size, device)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train', train_data, val_data, context_size, batch_size, device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(enc.decode(m.generate(context, max_new_tokens=500)[0].tolist()))