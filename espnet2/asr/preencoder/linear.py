#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder


class LinearProjection(AbsPreEncoder):
    """Linear Projection Preencoder."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        self.output_dim = output_size
        self.ff_1 = torch.nn.Linear(input_size, 512)
        self.ff_2 = torch.nn.Linear(512, output_size)
        self.activation = torch.nn.ReLU()
        #self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        output = self.activation(self.ff_1(input))
        output = self.activation(self.ff_2(output))
        return output, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
