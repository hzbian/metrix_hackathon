from typing import Tuple, Type, Any, Dict, List
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from einops.layers.torch import Rearrange


class MLP(nn.Module):
    """
    Simple MLP module. Note: the last layer has no activation function.
    """

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


class SurrogateBackbone(nn.Module, metaclass=ABCMeta):
    """
    Base class for backbones to be used in :class:`ray_nn.nn.models.SurrogateModel`.
    """

    @abstractmethod
    def forward(self,
                inp_params: torch.Tensor,
                inp_hist: torch.Tensor,
                inp_x_lims: torch.Tensor,
                inp_y_lims: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class TransformerBackbone(SurrogateBackbone):

    def __init__(self,
                 hist_dim: Tuple[int, int],  # x- and y-dim of histograms
                 n_hist_layers_inp: int,  # number of histograms in input plane or number of templates
                 n_hist_layers_out: int,  # number of histograms in output plane
                 param_dim: int,  # total number of parameters
                 transformer_dim: int = 1024,
                 transformer_mlp_dim: int = 2048,
                 transformer_heads: int = 4,
                 transformer_layers: int = 3,
                 use_inp_template: bool = False,  # use learnable template histograms instead of previous plane outputs
                 ):
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

        # Linear embedding layers for input data
        # c = number of n_hist_layers_inp / b = batch_size
        # note that the first transformer dimension is the sequences channel and the second the batch
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

        # total sequence length after concatenation of all embedded inputs
        transformer_channels = 1 + 3 * self.n_hist_layers_inp

        # create learnable templates
        if self.use_inp_template:
            self.inp_hist_template = nn.Parameter(torch.randn(self.n_hist_layers_inp, 1, self.transformer_dim),
                                                  requires_grad=True)
            self.inp_x_lims_template = nn.Parameter(torch.randn(self.n_hist_layers_inp, 1, self.transformer_dim),
                                                    requires_grad=True)
            self.inp_y_lims_template = nn.Parameter(torch.randn(self.n_hist_layers_inp, 1, self.transformer_dim),
                                                    requires_grad=True)

        # linear output layers to produce the final result
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
            # use input templates (no embeddings needed)
            inp_hist = self.inp_hist_template.expand(-1, bs, -1)
            inp_x_lims = self.inp_x_lims_template.expand(-1, bs, -1)
            inp_y_lims = self.inp_y_lims_template.expand(-1, bs, -1)
        else:
            # use input from previous plane
            inp_hist = self.to_hist_embedding(inp_hist)
            inp_x_lims = self.to_x_lims_embedding(inp_x_lims)
            inp_y_lims = self.to_y_lims_embedding(inp_y_lims)

        feat = self.transformer(torch.cat([inp_params, inp_hist, inp_x_lims, inp_y_lims], dim=0))

        out_hist = self.from_hist_embedding(feat)
        out_x_lims = self.from_x_lim_embedding(feat)
        out_y_lims = self.from_y_lim_embedding(feat)

        return out_hist, out_x_lims, out_y_lims


class CNNBackbone(SurrogateBackbone):
    """
    Backbone using conditional CNN blocks based on FiLM (https://distill.pub/2018/feature-wise-transformations/).
    """

    def __init__(self,
                 hist_dim: Tuple[int, int],
                 param_dim: int,
                 n_hist_layers: int,
                 n_templates: int = 32,
                 param_emb_n_layers: int = 5,
                 param_emb_dim_hidden: int = 128,
                 param_emb_dim: int = 1024,
                 lims_n_layers: int = 5,
                 lims_dim_hidden: int = 128,
                 conv_n_layers: int = 5,
                 conv_n_channels: int = 32,
                 conv_kernel_size: int = 3):
        super().__init__()

        self.hist_dim = hist_dim
        self.param_dim = param_dim
        self.n_hist_layers = n_hist_layers

        self.n_templates = n_templates
        self.templates = nn.Parameter(torch.randn(1, self.n_templates, self.hist_dim[0], self.hist_dim[1]),
                                      requires_grad=True)

        self.param_emb_dim = param_emb_dim
        self.param_emb_n_layers = param_emb_n_layers
        self.param_emb_dim_hidden = param_emb_dim_hidden
        self.param_embedder = MLP(dim_in=self.param_dim, dim_out=self.param_emb_dim,
                                  dim_hidden=(self.param_emb_n_layers - 1) * [self.param_emb_dim_hidden])

        self.conv_n_layers = conv_n_layers
        self.conv_n_channels = conv_n_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_blocks = nn.ModuleList([self._ConditionalConvBlock(param_emb_dim=self.param_emb_dim,
                                                                     conv_in_channels=self.n_templates,
                                                                     conv_out_channels=self.conv_n_channels,
                                                                     conv_kernel_size=self.conv_kernel_size)]
                                         + [self._ConditionalConvBlock(param_emb_dim=self.param_emb_dim,
                                                                       conv_in_channels=self.conv_n_channels,
                                                                       conv_out_channels=self.conv_n_channels,
                                                                       conv_kernel_size=self.conv_kernel_size)
                                            for _ in range(self.conv_n_layers - 1)])
        self.conv_out = nn.Conv2d(in_channels=self.conv_n_channels, out_channels=self.n_hist_layers, kernel_size=1)

        self.lims_n_layers = lims_n_layers
        self.lims_dim_hidden = lims_dim_hidden
        self.x_lims_predictor = MLP(dim_in=self.param_emb_dim, dim_out=2 * self.n_hist_layers,
                                    dim_hidden=(self.lims_n_layers - 1) * [self.lims_dim_hidden])
        self.y_lims_predictor = MLP(dim_in=self.param_emb_dim, dim_out=2 * self.n_hist_layers,
                                    dim_hidden=(self.lims_n_layers - 1) * [self.lims_dim_hidden])

    def forward(self, inp_params: torch.Tensor, *args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = inp_params.shape[0]
        param_emb = self.param_embedder(inp_params)

        feat = self.templates.expand(bs, -1, -1, -1)
        for conv_block in self.conv_blocks:
            feat = conv_block(feat, param_emb.clone())

        out_hist = self.conv_out(feat)
        out_x_lims = self.x_lims_predictor(param_emb).unsqueeze(1)
        out_y_lims = self.y_lims_predictor(param_emb).unsqueeze(1)

        return out_hist, out_x_lims, out_y_lims

    class _ConditionalConvBlock(nn.Module):

        def __init__(self,
                     param_emb_dim: int,
                     conv_in_channels: int,
                     conv_out_channels: int,
                     conv_kernel_size: int):
            super().__init__()

            self.conv1 = nn.Sequential(nn.Conv2d(in_channels=conv_in_channels, out_channels=conv_out_channels,
                                                 kernel_size=conv_kernel_size, padding='same'),
                                       nn.BatchNorm2d(conv_out_channels),
                                       nn.ReLU(True))
            self.conv2 = nn.Sequential(nn.Conv2d(in_channels=conv_out_channels, out_channels=conv_out_channels,
                                                 kernel_size=conv_kernel_size, padding='same'),
                                       nn.BatchNorm2d(conv_out_channels),
                                       nn.ReLU(True))
            self.conv3 = nn.Sequential(nn.Conv2d(in_channels=conv_out_channels, out_channels=conv_out_channels,
                                                 kernel_size=1),
                                       nn.BatchNorm2d(conv_out_channels),
                                       nn.ReLU(True))

            def _cond_template():
                return nn.Linear(param_emb_dim, conv_out_channels)

            self.cond_scal1 = _cond_template()
            self.cond_scal2 = _cond_template()
            self.cond_scal3 = _cond_template()

            self.cond_bias1 = _cond_template()
            self.cond_bias2 = _cond_template()
            self.cond_bias3 = _cond_template()

        def forward(self, feat: torch.Tensor, param_emb: torch.Tensor) -> torch.Tensor:
            feat = self.cond_scal1(param_emb).unsqueeze(-1).unsqueeze(-1) * self.conv1(feat) + \
                   self.cond_bias1(param_emb).unsqueeze(-1).unsqueeze(-1)
            feat = self.cond_scal2(param_emb).unsqueeze(-1).unsqueeze(-1) * self.conv2(feat) + \
                   self.cond_bias2(param_emb).unsqueeze(-1).unsqueeze(-1)
            feat = self.cond_scal3(param_emb).unsqueeze(-1).unsqueeze(-1) * self.conv3(feat) + \
                   self.cond_bias3(param_emb).unsqueeze(-1).unsqueeze(-1)
            return feat
