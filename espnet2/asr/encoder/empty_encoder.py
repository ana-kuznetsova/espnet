from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class EmptyEncoder(AbsEncoder):
    """EmptyEncoder class.

    Args:
        input_size: The number of expected features in the input
    """

    def __init__(
        self,
        input_size: int
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = input_size

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, FileNotFoundError]:

        return xs_pad, ilens, None
