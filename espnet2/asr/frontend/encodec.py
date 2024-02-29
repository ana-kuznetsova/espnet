from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from espnet2.asr.frontend.abs_frontend import AbsFrontend
import logging
from typing import List
import numpy as np
from encodec.model import EncodecModel
import re
from collections import OrderedDict


class EnCodecFrontend(AbsFrontend):
    '''EnCodec speech codec frontend.
    Args:
        fs: sampling frequency of the codec
        n_quantizers: desired number of quantizers for a specific bitrate
        trainable: freeze codec or not
    '''
    def __init__(self, model_path: str = "",
                       bandwidth: float = 6,
                       fs: float = 16000,
                       quantizer: bool = True,
                       ) -> None:
        super().__init__()
    
        model = EncodecModel._get_model(
                    target_bandwidths = [1.5, 3, 6], 
                    sample_rate = 16000,
                    channels  = 1,
                    causal  = True,
                    model_norm  = 'weight_norm',
                    audio_normalize  = False,
                    segment = None,
                    name = 'unset').cuda()
        state_dict = torch.load(model_path)
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = state_dict
        model.load_state_dict(model_dict)
        model.set_target_bandwidth(bandwidth)
        model.eval()
        
        self.encoder = model.encoder
        if quantizer:
            self.quantizer = model.quantizer
        del model
        self.feat_dim = 128
        self.fs = fs
        self.use_quantizer = quantizer

    def output_size(self) -> int:
        return self.feat_dim

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.unsqueeze(1)
        z = self.encoder(input)
        if self.use_quantizer:
            quantizedResult = self.quantizer(z, frame_rate=16000) 
            z = quantizedResult.quantized
        #Dont forget to calculate lens
        bsize, feat_dim, length = z.size()
        z = z.contiguous().view(bsize, length, feat_dim)
        input_lengths = torch.Tensor([length] * bsize).long().to(z.device)
        commitment_loss, codebook_loss = None, None
        return z, input_lengths, commitment_loss, codebook_loss

    
