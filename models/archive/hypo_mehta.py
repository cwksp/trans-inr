import torch
import torch.nn as nn
import numpy as np

from models import register
from models.hyponets.layers import batched_linear_mm


@register('hypo_mehta')
class HypoMehta(nn.Module):

    def __init__(self, depth, in_dim, out_dim, hidden_dim, use_pe, pe_dim):
        super().__init__()
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.depth = depth
        self.param_shapes = dict()
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim)
            last_dim = cur_dim
        self.relu = nn.ReLU()
        self.params = None

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = 2**torch.linspace(0, 10, self.pe_dim // 2, device=x.device)
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def forward(self, x):
        B, query_shape = x.shape[0], x.shape[1: -1]
        x = x.view(B, -1, x.shape[-1])
        if self.use_pe:
            x = self.convert_posenc(x)
        for i in range(self.depth):
            name = f'wb{i}'
            x = batched_linear_mm(x, self.params[name])
            if i < self.depth - 1:
                x = self.relu(x) * self.params['_' + name + '_alpha'].unsqueeze(1)
            else:
                x = x * self.params['_' + name + '_alpha'].unsqueeze(1) + 0.5 ##
        x = x.view(B, *query_shape, -1)
        return x
