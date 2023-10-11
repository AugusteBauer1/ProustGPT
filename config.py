import torch

batch_size = 32 
context_size = 128
max_iters = 5000
eval_interval = 300
learning_rate = 31e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
number_embed = 384
number_head = 6
inner_linear_factor = 4
n_block = 6
dropout = 0.2