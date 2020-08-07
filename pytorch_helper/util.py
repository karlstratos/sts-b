import numpy as np
import torch
import torch.nn as nn


def masked_mean(hiddens, mask):  # B x T x d, B x T
    B, T, d = hiddens.size()
    lengths = mask.sum(dim=1, keepdim=True)  # B x 1
    mask = mask.unsqueeze(2).expand(B, T, d)
    mean = (mask * hiddens).sum(dim=1).true_divide(lengths)
    return mean  # B x T


def get_length_mask(lengths, max_length=None):
    if not max_length:
        max_length = lengths.max().item()
    indices = torch.arange(max_length).expand(len(lengths), -1)
    lengths = lengths.unsqueeze(1).expand(len(lengths), max_length)
    mask = indices < lengths
    return mask  # B x T


def get_init_uniform(init_value):

    def init_uniform(m):
        if init_value > 0.:  # If 0 use default initialization
            if hasattr(m, 'weight'):
                m.weight.data.uniform_(-init_value, init_value)
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0.)

    return init_uniform

def get_init_transformer(transformer):
    """
    Initialization scheme used for transformers:
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
    """
    def init_transformer(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=transformer.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    return init_transformer


class FF(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers,
                 activation='relu', dropout_rate=0, layer_norm=False,
                 residual_connection=False):
        super().__init__()

        assert num_layers >= 0  # 0 = Linear
        if num_layers > 0:
            assert dim_hidden > 0
        if residual_connection:
            assert dim_hidden == dim_input

        self.residual_connection = residual_connection
        self.stack = nn.ModuleList()
        for l in range(num_layers):
            layer = []

            if layer_norm:
                layer.append(nn.LayerNorm(dim_input if l == 0 else dim_hidden))

            layer.append(nn.Linear(dim_input if l == 0 else dim_hidden,
                                   dim_hidden))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[activation])

            if dropout_rate > 0:
                layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_input if num_layers < 1 else dim_hidden,
                             dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = x + layer(x) if self.residual_connection else layer(x)
        return self.out(x)


def cos_numpy(u, v):
    return np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
