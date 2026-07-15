import torch
import torch.nn as nn

d_in = 3
d_out = 2

torch.manual_seed(123)
W_query = nn.Parameter(torch.rand(d_in, d_out))
torch.manual_seed(123)
W_key = nn.Parameter(torch.rand(d_in, d_out))
torch.manual_seed(123)
W_value = nn.Parameter(torch.rand(d_in, d_out))

print(W_query, W_key, W_value)