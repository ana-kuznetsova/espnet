"""SVD post encoder for reducing data rate."""

from typing import Optional, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder

class SVDPostencoder(AbsPostEncoder):
    def __init__(
        self,
        input_size: int,
        svd_dim:int, 
        output_size: Optional[int] = None, 
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()
        self.input_size = input_size
        self.svd_dim = svd_dim
        self.out_size = output_size
    
    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor]:
        U, S, VT = torch.linalg.svd(input)
        S = torch.diag(S)
        output = U[:, :self.svd_dim] @ S[:self.svd_dim, :self.svd_dim] @ VT[:self.svd_dim]
        return output, input_lengths

    def output_size(self) -> int:
        return self.out_size