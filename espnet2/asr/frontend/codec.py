from typing import Optional, Tuple, Union
import torch
import dac
from espnet2.asr.frontend.abs_frontend import AbsFrontend

class CodecFrontend(AbsFrontend):
    '''DAC speech codec frontend.
    Args:
        fs: sampling frequency of the codec
        n_quantizers: desired number of quantizers for a specific bitrate
        trainable: freeze codec or not
    '''
    def __init__(self, fs: int = 16000,
                       n_quantizers: int = 6,
                       trainable: bool = False,
                       normalize_codes: bool = False) -> None:
        super().__init__()
        self.fs = fs
        self.feat_dim = 1024
        self.n_quantizers = n_quantizers
        codec_path = dac.utils.download(model_type="16khz")
        self.codec = dac.DAC.load(codec_path)
        if trainable:
            self.codec.train()
        else:
            self.codec.eval()
        self.normalize_codes = normalize_codes

    def output_size(self) -> int:
        return self.feat_dim

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #print("Inp len", input_lengths)
        input = input.unsqueeze(1)
        z, _, _, _, _ = self.codec.encode(input, self.n_quantizers)
        # Convert input to (B, L, Dim)
        bsize, feat_dim, length = z.size()
        z = z.view(bsize, length, feat_dim)
        #Repeat each frame twice to match the original MFCC framerate
        z = z.repeat_interleave(2, dim=1)
        input_lengths = torch.Tensor([length * 2] * bsize)
        print("DEBUG Z", z.shape)
        if self.normalize_codes:
            max_val = z.max(dim=1)
            print(max_val.values.shape)
            z = z/max_val.values
        #print("Inp lens out", bsize, input_lengths)
        return z, input_lengths