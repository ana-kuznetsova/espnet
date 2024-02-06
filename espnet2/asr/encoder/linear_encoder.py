from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import check_argument_types
import logging

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class LinearEncoder(AbsEncoder):
    """LinearEncoder class.

    Args:
        input_size: The number of expected features in the input
        output_size: The number of output features
        num_layers: Number of recurrent layers
        dropout: dropout probability

    """

    def __init__(
        self,
        input_size: int,
        num_layers: int = 1,
        output_size: int = 256,
        residual: bool = False,
        factor: bool = False,
        dropout: Optional[float]= 0.0,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.residual = residual

        if num_layers > 1 and factor:
            # Calculate hidden dims depending on the output_size
            # and num_layers
            assert input_size % output_size == 0 and input_size % num_layers == 0
            hidden_dims = []

            for i in range(num_layers):
                if i == 0:
                    hidden_dims.append((input_size, input_size // 2))
                else:
                    latest_dim = hidden_dims[i-1][1]
                    hidden_dims.append((latest_dim, latest_dim // 2))
            assert len(hidden_dims) == num_layers and hidden_dims[-1][1] == output_size

        else:
            if num_layers == 1:
                hidden_dims = [(input_size, output_size)]
            else:
                hidden_dims = []
                for i in range(num_layers):
                    if i == 0:
                        hidden_dims.append((input_size, input_size // 2))
                    elif i == num_layers - 1:
                        hidden_dims.append((input_size // 2, output_size))
                    else:
                        hidden_dims.append((input_size // 2, input_size // 2))

        all_layers = []
        for i in range(num_layers):
            if not residual:
                all_layers.append(torch.nn.Linear(in_features=hidden_dims[i][0], 
                                                out_features=hidden_dims[i][1]))
                all_layers.append(torch.nn.SiLU())

                if dropout:
                    dropout_layer = torch.nn.Dropout(p=dropout)
                    all_layers.append(dropout_layer)
            else:
                all_layers.append(torch.nn.Linear(in_features=hidden_dims[i][0], 
                                                    out_features=hidden_dims[i][1]))
        
        self.enc = torch.nn.ModuleList(all_layers)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, FileNotFoundError]:
        
        for module in self.enc:
            if not self.residual:
                xs_pad = module(xs_pad)
            else:
                xs_pad = module(xs_pad)
                
                xs_pad = xs_pad + (0.5 * F.silu(xs_pad))
        return xs_pad, ilens, None
