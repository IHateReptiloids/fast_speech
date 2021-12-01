from positional_encodings import PositionalEncoding1D
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs import FastSpeechConfig
from .self_attention import SelfAttention


class FastSpeech(nn.Module):
    def __init__(self, config: FastSpeechConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.n_tokens, config.d_model)
        self.pos_encoding = PositionalEncoding1D(config.d_model)
        self.phoneme_layers = nn.Sequential(*[
            FFTBlock(config)
            for _ in range(config.n_layers)
        ])
        self.length_regulator = LengthRegulator(config)
        self.mel_layers = nn.Sequential(*[
            FFTBlock(config)
            for _ in range(config.n_layers)
        ])
        self.projector = nn.Linear(config.d_model, config.n_mels)
        self.to(config.device)

    def forward(self, x, y=None):
        '''
        x is encoded text of shape (bs, seq_len)
        y: target lengths
        if y is not None, computes loss between predicted
        lengths and target lengths
        returns: mel specs of shape (bs, n_mels, time)
        '''
        x = self.embeddings(x)
        x = x + self.pos_encoding(x)
        x = self.phoneme_layers(x)
        x = self.length_regulator(x, y)
        x = x + self.pos_encoding(x)
        x = self.mel_layers(x)
        x = self.projector(x)
        return x.transpose(-1, -2)


class LengthRegulator(nn.Module):
    def __init__(self, config: FastSpeechConfig):
        super().__init__()
        self.duration_predictor = DurationPredictor(config)
        self._loss = None

    def forward(self, x, y=None):
        '''
        x is of shape (bs, seq_len, d_model)
        y: target lengths
        if y is not None, computes loss between predicted
        lengths and target lengths
        '''
        lengths = self.duration_predictor(x)
        assert lengths.shape == x.shape[:-1]
        if y is not None:
            self._loss = F.mse_loss(lengths, y)
            lengths = y.round().int()
        else:
            lengths = lengths.round().int()
        repeated = []
        for seq, seq_lengths in zip(x, lengths):
            repeated.append(torch.repeat_interleave(seq, seq_lengths, dim=0))
        return nn.utils.rnn.pad_sequence(repeated, batch_first=True)

    @property
    def loss(self):
        _loss = self._loss
        self._loss = None
        return _loss


class DurationPredictor(nn.Module):
    '''
    predicts log lengths
    '''
    def __init__(self, config):
        super().__init__()

        layers = []
        d_in, d_out = config.d_model, config.d_dp
        for _ in range(2):
            layers.append(TConvT(d_in, d_out, config.dp_kernel_size,
                                 padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(d_out))
            layers.append(nn.Dropout(config.dropout))
            d_in = d_out
        layers.append(nn.Linear(d_out, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.exp(self.net(x)).squeeze(dim=-1)


class FFTBlock(nn.Module):
    def __init__(self, config: FastSpeechConfig):
        super().__init__()
        attention = ResidualAndNorm(
            SelfAttention(config.d_model, config.d_model, config.n_heads),
            config.d_model, config.dropout
        )
        conv = ResidualAndNorm(
            nn.Sequential(
                TConvT(config.d_model, config.d_ff,
                       config.ff_kernel_size[0], padding='same'),
                nn.ReLU(),
                TConvT(config.d_ff, config.d_model,
                       config.ff_kernel_size[1], padding='same')
            ),
            config.d_model, config.dropout
        )
        self.net = nn.Sequential(attention, conv)

    def forward(self, x):
        return self.net(x)


class ResidualAndNorm(nn.Module):
    def __init__(self, layer, d_model, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            layer,
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layer_norm(x + self.layer(x))


class TConvT(nn.Conv1d):
    def forward(self, x):
        return super().forward(x.transpose(-1, -2)).transpose(-1, -2)
