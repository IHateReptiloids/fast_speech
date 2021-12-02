from pathlib import Path
import string

import numpy as np
import torch
import torchaudio


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root='data/lj_speech', indices_path=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        super().__init__(root=root, download=True)
        self._tokenizer = torchaudio.pipelines \
            .TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self._indices = None
        if indices_path is not None:
            self._indices = np.loadtxt(indices_path, dtype=np.int32)

    def __getitem__(self, index: int):
        if self._indices is not None:
            index = self._indices[index]
        waveform, _, _, transcript = super().__getitem__(index)

        to_remove = string.punctuation.replace("'", '')
        transcript = transcript.replace('"', "'")
        transcript = transcript.translate(str.maketrans('', '', to_remove))

        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)
        assert token_lengths.item() == len(transcript), transcript

        return waveform, waveform_length, transcript, tokens, token_lengths

    def __len__(self):
        if self._indices is not None:
            return len(self._indices)
        return super().__len__()

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
