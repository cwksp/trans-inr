import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import models
from models import register


def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()


@register('trans_mehta')
class TransMehta(nn.Module):

    def __init__(self, tokenizer, hyponet, transformer_encoder):
        super().__init__()
        dim = transformer_encoder['args']['dim']
        self.tokenizer = models.make(tokenizer, args={'dim': dim})
        self.hyponet = models.make(hyponet)
        self.transformer_encoder = models.make(transformer_encoder)

        self.base_params = nn.ParameterDict()
        self.modnet = nn.ModuleDict()
        last_dim = 0

        for i in range(self.hyponet.depth):
            name = f'wb{i}'
            shape = self.hyponet.param_shapes[name]
            self.base_params[name] = nn.Parameter(init_wb(shape))
            self.modnet[name] = nn.Linear(last_dim + dim, shape[1])
            last_dim = shape[1]

        self.wtokens = nn.Parameter(torch.randn(1, dim))

    def forward(self, data):
        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1))
        z = trans_out[:, -1, :]

        params = dict()
        h_last = None
        for i in range(self.hyponet.depth):
            name = f'wb{i}'
            shape = self.hyponet.param_shapes[name]
            params[name] = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)

            if i == 0:
                x = z
            else:
                x = torch.cat([h_last, z], dim=1)
            h_last = self.modnet[name](x)
            params['_' + name + '_alpha'] = h_last

        self.hyponet.set_params(params)
        return self.hyponet
