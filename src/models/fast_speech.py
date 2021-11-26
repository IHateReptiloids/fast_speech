from positional_encodings import PositionalEncoding1D
import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attention import SelfAttention


class FastSpeech(nn.Module):
    def __init__(
        self,
        n_tokens,
        n_mels,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        kernel_size,
        dropout
    ):
        super().__init__()
        self.embeddings = nn.Embedding(n_tokens, d_model)
        self.pos_encoding = PositionalEncoding1D(d_model)
        self.phoneme_layers = nn.Sequential(*[
            FFTBlock(d_model, d_ff, n_heads, kernel_size, dropout)
            for _ in range(n_layers)
        ])
        self.length_regulator = LengthRegulator(d_model, kernel_size, dropout)
        self.mel_layers = nn.Sequential(*[
            FFTBlock(d_model, d_ff, n_heads, kernel_size, dropout)
            for _ in range(n_layers)
        ])
        self.projector = nn.Linear(d_model, n_mels)

    def forward(self, x, y=None):
        '''
        x is encoded text of shape (bs, seq_len)
        y: target lengths
        if y is not None, computes loss between predicted
        lengths and target lengths
        '''
        x = self.embeddings(x)
        x = self.pos_encoding(x)
        x = self.phoneme_layers(x)
        x = self.length_regulator(x, y)
        x = self.mel_layers(x)
        x = self.projector(x)
        return x


class LengthRegulator(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.duration_predictor(d_model, kernel_size, dropout)
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
            assert y.dtype == torch.int32
            self._loss = F.mse_loss(lengths, y.float())
            lengths = y
        else:
            lengths = lengths.int()
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
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        layers = []
        for _ in range(2):
            layers.append(nn.Conv1d(d_model, d_model, kernel_size,
                                    padding='same'))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(d_model))
            layers.append(nn.Dropout(dropout, inplace=True))
        layers.append(nn.Linear(d_model, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.exp(self.net(x)).squeeze()


class FFTBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, kernel_size, dropout):
        super().__init__()
        conv = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_ff, d_model, kernel_size, padding='same')
        )
        self.net = nn.Sequential(
            ResidualAndNorm(SelfAttention(d_model, d_model, n_heads),
                            d_model, dropout),
            ResidualAndNorm(conv, d_model, dropout)
        )

    def forward(self, x):
        return self.net(x)


class ResidualAndNorm(nn.Module):
    def __init__(self, layer, d_model, dropout=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            layer,
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layer_norm(x + self.layer(x))
