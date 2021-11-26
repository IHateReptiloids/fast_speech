import torch
from torch.nn.utils.rnn import pad_sequence

from .batch import Batch


def collate(instances):
    waveform, waveform_length, transcript, tokens, token_lengths = list(
        zip(*instances)
    )

    waveform = pad_sequence([
        waveform_[0] for waveform_ in waveform
    ]).transpose(0, 1)
    waveform_length = torch.cat(waveform_length)

    tokens = pad_sequence([
        tokens_[0] for tokens_ in tokens
    ]).transpose(0, 1)
    token_lengths = torch.cat(token_lengths)

    return Batch(waveform, waveform_length, transcript, tokens, token_lengths)
