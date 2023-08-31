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
        print("DEBUG INPUT", input.shape)
        num_elements = input.shape[1]
        input_lengths = torch.Tensor([self.feat_dim] * num_elements)
        input = input.view(1, 1, -1)
        assert len(input.size()) == 3, "Input should be of shape (B, C, L)"
        z, _, _, _, _ = self.codec.encode(input)
        return z, input_lengths