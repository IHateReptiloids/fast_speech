from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class FastSpeechConfig:
    # MelSpectrogram params
    sample_rate: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 11000
    n_mels: int = 80
    power: float = 1.0
    # FastSpeech params
    n_layers: int = 6
    d_model: int = 384
    d_ff: int = 1536
    d_dp: int = 256
    n_heads: int = 2
    dp_kernel_size: int = 3
    ff_kernel_size: Tuple[int, int] = (3, 3)
    dropout: float = 0.1
    # Adam params
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    initial_lr: float = 1.0
    warmup_steps: int = 4000
    weight_decay: float = 0.0
    # Other
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available()
                                        else 'cpu')
    n_tokens: int = 38
    num_epochs: int = 3000


@dataclass
class FastSpeech2Config(FastSpeechConfig):
    n_layers: int = 4
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 2
    ff_kernel_size: Tuple[int, int] = (9, 1)
