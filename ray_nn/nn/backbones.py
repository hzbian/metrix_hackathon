from typing import Tuple, Type, Any, Dict, List
from collections import OrderedDict

import torch
from torch import nn
from einops.layers.torch import Rearrange


class MLP(nn.Module):

    def __init__(self,
                 dim_in: int, dim_out: int, dim_hidden: List[int],
                 activation: Type[Any] = nn.ReLU,
                 activation_params: Dict = dict(inplace=True)):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.activation = activation(**activation_params)

        dims = [self.dim_in] + self.dim_hidden + [self.dim_out]
        self.depth = len(dims) - 1

        modules = []
        for idx in range(1, len(dims)):
            modules += [('layer' + str(idx), nn.Linear(dims[idx - 1], dims[idx], bias=True)),
                        ('act' + str(idx), self.activation)]
        modules.pop()  # no final activation
        self.net = nn.Sequential(OrderedDict(modules))

    def forward(self, x: torch.Tensor):
        return self.net(x.view(x.shape[0], -1))


class TransformerBackbone(nn.Module):

    def __init__(self,
                 hist_dim: Tuple[int, int],
                 n_hist_layers_inp: int,
                 n_hist_layers_out: int,
                 param_dim: int,
                 transformer_dim: int = 1024,
                 transformer_mlp_dim: int = 2048,
                 transformer_heads: int = 4,
                 transformer_layers: int = 3,
                 use_inp_template: bool = False):
        super().__init__()

        self.hist_dim = hist_dim
        self.n_hist_layers_inp = n_hist_layers_inp
        self.n_hist_layers_out = n_hist_layers_out
        self.param_dim = param_dim
        self.transformer_dim = transformer_dim
        self.transformer_mlp_dim = transformer_mlp_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.use_inp_template = use_inp_template

        _encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim,
                                                    nhead=self.transformer_heads,
                                                    dim_feedforward=self.transformer_mlp_dim)
        self.transformer = nn.TransformerEncoder(_encoder_layer, num_layers=self.transformer_layers)

        self.to_param_embedding = nn.Sequential(
            Rearrange('b p -> 1 b p'),
            nn.Linear(self.param_dim, self.transformer_dim))
        self.to_hist_embedding = nn.Sequential(
            Rearrange('b c x y -> c b (x y)'),
            nn.Linear(self.hist_dim[0] * self.hist_dim[1], self.transformer_dim))
        self.to_x_lims_embedding = nn.Sequential(
            Rearrange('b c x -> c b x'),
            nn.Linear(2, self.transformer_dim))
        self.to_y_lims_embedding = nn.Sequential(
            Rearrange('b c y -> c b y'),
            nn.Linear(2, self.transformer_dim))

        transformer_channels = 1 + 3 * self.n_hist_layers_inp

        if self.use_inp_template:
            self.inp_hist_template = nn.Parameter(torch.randn(self.n_hist_layers_inp, 1, self.transformer_dim),
                                                  requires_grad=True)
            self.inp_x_lims_template = nn.Parameter(torch.randn(self.n_hist_layers_inp, 1, self.transformer_dim),
                                                    requires_grad=True)
            self.inp_y_lims_template = nn.Parameter(torch.randn(self.n_hist_layers_inp, 1, self.transformer_dim),
                                                    requires_grad=True)

        self.from_hist_embedding = nn.Sequential(
            Rearrange('c b d -> b c d'),
            nn.Conv1d(in_channels=transformer_channels, out_channels=self.n_hist_layers_out, kernel_size=1),
            nn.Linear(self.transformer_dim, self.hist_dim[0] * self.hist_dim[1]),
            Rearrange('b c (x y) -> b c x y', x=self.hist_dim[0]),
            # nn.Conv2d(in_channels=self.n_hist_layers_out, out_channels=self.n_hist_layers_out,
            #           kernel_size=3, padding='same'),
            # nn.Conv2d(in_channels=self.n_hist_layers_out, out_channels=self.n_hist_layers_out,
            #           kernel_size=3, padding='same'),
        )
        self.from_x_lim_embedding = nn.Sequential(
            Rearrange('c b d -> b c d'),
            nn.Conv1d(in_channels=transformer_channels, out_channels=self.n_hist_layers_out, kernel_size=1),
            nn.Linear(self.transformer_dim, 2))
        self.from_y_lim_embedding = nn.Sequential(
            Rearrange('c b d -> b c d'),
            nn.Conv1d(in_channels=transformer_channels, out_channels=self.n_hist_layers_out, kernel_size=1),
            nn.Linear(self.transformer_dim, 2))

    def forward(self,
                inp_params: torch.Tensor,
                inp_hist: torch.Tensor,
                inp_x_lims: torch.Tensor,
                inp_y_lims: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = inp_params.shape[0]
        inp_params = self.to_param_embedding(inp_params)
        if self.use_inp_template:
            inp_hist = self.inp_hist_template.expand(-1, bs, -1)
            inp_x_lims = self.inp_x_lims_template.expand(-1, bs, -1)
            inp_y_lims = self.inp_y_lims_template.expand(-1, bs, -1)
        else:
            inp_hist = self.to_hist_embedding(inp_hist)
            inp_x_lims = self.to_x_lims_embedding(inp_x_lims)
            inp_y_lims = self.to_x_lims_embedding(inp_y_lims)

        # TODO: cat triples of hist, x_lims, y_lims
        feat = self.transformer(torch.cat([inp_params, inp_hist, inp_x_lims, inp_y_lims], dim=0))

        out_hist = self.from_hist_embedding(feat)
        out_x_lims = self.from_x_lim_embedding(feat)
        out_y_lims = self.from_y_lim_embedding(feat)

        return out_hist, out_x_lims, out_y_lims


class TemplateBackbone(nn.Module):

    def __init__(self,
                 hist_dim: Tuple[int, int],
                 param_dim: int,
                 n_hist_layers_out: int,
                 n_templates: int = 64,
                 param_emb_n_layers: int = 5,
                 param_emb_dim_hidden: int = 128,
                 param_emb_dim: int = 1024,
                 conv_n_layers: int = 5,
                 conv_n_channels: int = 32,
                 conv_kernel_size: int = 3):
        super().__init__()

        self.hist_dim = hist_dim
        self.param_dim = param_dim

        self.n_hist_layers_out = n_hist_layers_out
        self.n_templates = n_templates

        self.param_emb_n_layers = param_emb_n_layers
        self.param_emb_dim_hidden = param_emb_dim_hidden
        self.param_emb_dim = param_emb_dim

        self.conv_n_layers = conv_n_layers
        self.conv_n_channels = conv_n_channels
        self.conv_kernel_size = conv_kernel_size

        self.templates = nn.Parameter(torch.randn(1, self.n_templates, self.hist_dim[0], self.hist_dim[1]),
                                      requires_grad=True)

        self.param_emb = MLP(dim_in=self.param_dim, dim_out=self.param_emb_dim,
                             dim_hidden=(self.param_emb_n_layers - 1) * [self.param_emb_dim_hidden])

    def forward(self,
                inp_params: torch.Tensor,
                inp_hist: torch.Tensor,
                inp_x_lims: torch.Tensor,
                inp_y_lims: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    class _ConditionalConvBlock(nn.Module):

        def __init__(self,
                     param_emb_dim: int,
                     conv_in_channels: int,
                     conv_out_channels: int,
                     conv_kernel_size: int):
            super().__init__()

            self.param_emb_dim = param_emb_dim
            self.conv_in_channels = conv_in_channels
            self.conv_out_channels = conv_out_channels
            self.conv_kernel_size = conv_kernel_size

        def forward(self, inp_feat: torch.Tensor, inp_params: torch.Tensor) -> torch.Tensor:
            pass
