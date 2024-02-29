from typing import Optional, Tuple, Union
import torch
import math
import torch.nn.functional as F
import dac
from espnet2.asr.frontend.abs_frontend import AbsFrontend
import logging
from typing import List
import numpy as np


class CodecFrontend(AbsFrontend):
    '''DAC speech codec frontend.
    Args:
        fs: sampling frequency of the codec
        n_quantizers: desired number of quantizers for a specific bitrate
        trainable: freeze codec or not
    '''
    def __init__(self, fs: int = 16000,
                       n_quantizers: int = 6,
                       normalize_codes: bool = False,
                       quantizer: bool = True,
                       preprocess_signal: bool = False,
                       layer_norm: bool = True,
                       encoder_rates: List[int] = [2, 4, 8, 8]) -> None:
        super().__init__()
        self.fs = fs
        self.feat_dim = 1024
        self.n_quantizers = n_quantizers
        self.normalize_codes = normalize_codes
        self.hop_length = np.prod(encoder_rates)
        self.preprocess_signal = preprocess_signal
        self.use_quantizer = quantizer
        self.use_layer_norm = layer_norm

        codec_path = dac.utils.download(model_type="16khz")
        model = dac.DAC.load(codec_path)

        self.encoder = model.encoder
        if self.use_quantizer:
            self.quantizer = model.quantizer

    def output_size(self) -> int:
        return self.feat_dim
    
    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.fs
        assert sample_rate == self.fs

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = F.pad(audio_data, (0, right_pad))

        return audio_data

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.unsqueeze(1)
        if self.preprocess_signal:
            input = self.preprocess(input, self.fs)


        z = self.encoder(input)
        bsize, feat_dim, length = z.size()
        if self.use_layer_norm:
            z = F.layer_norm(z, normalized_shape=[bsize, feat_dim, length])
        #z_q, commitment_loss, codebook_loss, indices, z_e
        if self.use_quantizer:
            z, commitment_loss , codebook_loss, _, _ = self.quantizer(z, self.n_quantizers)

        # Convert input to (B, L, Dim)
        bsize, feat_dim, length = z.size()
        z = z.view(bsize, length, feat_dim)
        input_lengths = torch.Tensor([length] * bsize).long().to(z.device)

        if self.use_quantizer:
            bsize, feat_dim, length = commitment_loss.size()
            commitment_loss = commitment_loss.view(bsize, length, feat_dim).float().mean([1, 2])
            
            bsize, feat_dim, length = codebook_loss.size()
            codebook_loss = codebook_loss.view(bsize, length, feat_dim).float().mean([1, 2])
        else:
            codebook_loss, commitment_loss = None, None

        return z, input_lengths, commitment_loss, codebook_loss