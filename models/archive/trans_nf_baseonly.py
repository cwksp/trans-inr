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


@register('trans_nf_baseonly')
class TransNfBaseonly(nn.Module):

    def __init__(self, hyponet):
        super().__init__()
        self.hyponet = models.make(hyponet)

        self.base_params = nn.ParameterDict()
        for name, shape in self.hyponet.param_shapes.items():
            self.base_params[name] = nn.Parameter(init_wb(shape))

    def forward(self, data):
        B = data['support_imgs'].shape[0]

        params = dict()
        for name, shape in self.hyponet.param_shapes.items():
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            params[name] = wb

        self.hyponet.set_params(params)
        return self.hyponet
