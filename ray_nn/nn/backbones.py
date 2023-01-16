from typing import Tuple

import torch
from torch import nn
from einops.layers.torch import Rearrange


class TransformerBackbone(nn.Module):

    def __init__(self,
                 hist_dim: Tuple[int, int],
                 n_hist_layers_inp: int,
                 n_hist_layers_out: int,
                 param_dim: int,
                 transformer_dim: int = 1024,
                 transformer_mlp_dim: int = 2048,
                 transformer_heads: int = 4,
                 transformer_layers: int = 3):
        super().__init__()

        self.hist_dim = hist_dim
        self.n_hist_layers_inp = n_hist_layers_inp
        self.n_hist_layers_out = n_hist_layers_out
        self.param_dim = param_dim
        self.transformer_dim = transformer_dim
        self.transformer_mlp_dim = transformer_mlp_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers

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

        self.from_hist_embedding = nn.Sequential(
            Rearrange('c b d -> b c d'),
            nn.Conv1d(in_channels=transformer_channels, out_channels=self.n_hist_layers_out, kernel_size=1),
            nn.Linear(self.transformer_dim, self.hist_dim[0] * self.hist_dim[1]),
            Rearrange('b c (x y) -> b c x y', x=self.hist_dim[0]))
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
        inp_params = self.to_param_embedding(inp_params)
        inp_hist = self.to_hist_embedding(inp_hist)
        inp_x_lims = self.to_x_lims_embedding(inp_x_lims)
        inp_y_lims = self.to_x_lims_embedding(inp_y_lims)

        # TODO: cat triples of hist, x_lims, y_lims
        feat = self.transformer(torch.cat([inp_params, inp_hist, inp_x_lims, inp_y_lims], dim=0))

        out_hist = self.from_hist_embedding(feat)
        out_x_lims = self.from_x_lim_embedding(feat)
        out_y_lims = self.from_y_lim_embedding(feat)

        return out_hist, out_x_lims, out_y_lims
