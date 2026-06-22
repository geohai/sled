from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from rshf.satclip.model import Siren, exists, cast_tuple
from rshf.geoclip.model import LocationEncoderCapsule
from torch import Tensor
from einops import rearrange

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


def equal_earth_projection(L):
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (
        2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)
    ) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180


# UniGeoCLIP's implementation with HuggingFace Hub mixin for easy weight loading/saving
class LocationEncoderUniGeoCLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        sigma=[2**0, 2**4, 2**8, 2**12],
        embed_dim=768,
        n_registers=4,
        num_heads=12,
        mlp_ratio=4,
        depth=12,
    ):
        super(LocationEncoderUniGeoCLIP, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        self.n_registers = n_registers

        self.loc_enc = nn.ModuleList(
            [
                GaussianEncoding(sigma=sigma, input_size=2, encoded_size=embed_dim // 2)
                for sigma in self.sigma
            ]
        )

        self.registers = nn.Parameter(torch.empty(n_registers, embed_dim))
        nn.init.normal_(self.registers, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )

    def forward(self, location):
        location = equal_earth_projection(location)
        location_features = []

        for gaussian_encoding in self.loc_enc:
            location_features.append(gaussian_encoding(location))

        location_features = torch.stack(location_features, dim=1)
        registers = self.registers.unsqueeze(0).expand(
            location_features.shape[0], -1, -1
        )
        tokens = torch.cat([registers, location_features], dim=1)

        tokens = self.transformer(tokens)

        return tokens[:, self.n_registers :, :].mean(dim=1)


class GaussianEncoding(nn.Module):
    """Map coordinates to a Gaussian Random Fourier Features embedding."""

    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[int] = None,
        encoded_size: Optional[int] = None,
    ):
        super().__init__()
        b = torch.randn(int(encoded_size), int(input_size)) * float(sigma)
        self.mat = nn.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        vp = 2 * np.pi * v @ self.mat.T
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


# RSHF's SphericalHarmonics implementation, but with HuggingFace Hub mixin for easy weight loading/saving
class LocationEncoderSatCLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self, position_encoding, nnet):
        super().__init__()
        self.posenc = position_encoding
        self.nnet = nnet

    def forward(self, x):
        x = self.posenc(x)
        return self.nnet(x)


# RSHF's SirenNet implementation, but with HuggingFace Hub mixin for easy weight loading/saving
class SirenNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, degreeinput = False, dropout = True):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.degreeinput = degreeinput

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation, dropout = False)

    def forward(self, x, mods = None):

        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)

# RSHF's GeoCLIP implementation, upgraded to handle how the transformers' library passes around configs
# for easy loading/saving in huggingface
class LocationEncoderGeoCLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self, sigma=None, input_size=2, encoded_size=256, dim=512):
        super(LocationEncoderGeoCLIP, self).__init__()
        self.sigma = sigma
        self.input_size = input_size
        self.encoded_size = encoded_size
        self.dim = dim
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s, input_size=self.input_size,
                                                                      encoded_size=self.encoded_size,
                                                                      dim=self.dim))

    def forward(self, location):
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], self.dim).to(location.device)

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)

        return location_features