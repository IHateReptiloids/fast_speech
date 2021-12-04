import torch
from torch.nn.utils.rnn import pad_sequence

from .batch import Batch


def collate(instances):
    waveform, waveform_length, transcript, tokens, token_lengths, durations = (
        list(zip(*instances))
    )

    waveform = pad_sequence([waveform_ for waveform_ in waveform],
                            batch_first=True)
    waveform_length = torch.cat(waveform_length)

    tokens = pad_sequence([tokens_ for tokens_ in tokens], batch_first=True)
    token_lengths = torch.cat(token_lengths)

    return Batch(waveform, waveform_length, transcript,
                 tokens, token_lengths, durations)
