from typing import Optional, Tuple, Union
import torch
import dac
from espnet2.asr.frontend.abs_frontend import AbsFrontend

class CodecFrontend(AbsFrontend):
    '''DAC speech codec frontend.
    '''
    def __init__(self, fs: int = 16000,
                       trainable: bool = False) -> None:
        super().__init__()
        self.fs = fs
        self.feat_dim = 1024
        codec_path = dac.utils.download(model_type="16khz")
        self.codec = dac.DAC.load(codec_path)
        if trainable:
            self.codec.train()
        else:
            self.codec.eval()

    def output_size(self) -> int:
        return self.feat_dim

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #print("Inp len", input_lengths)
        input = input.view(1, 1, -1)
        z, _, _, _, _ = self.codec.encode(input)
        # Convert input to (B, L, Dim)
        bsize, feat_dim, length = z.size()
        z = z.view(bsize, length, feat_dim)
        input_lengths = torch.Tensor([length] * bsize)
        #print("Inp lens out", bsize, input_lengths)
        return z, input_lengths