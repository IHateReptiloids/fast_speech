from pathlib import Path

import numpy as np
import torch
import torchaudio


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, aligner, root='data/lj_speech', indices_path=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        super().__init__(root=root, download=True)
        self._aligner = aligner
        self._tokenizer = torchaudio.pipelines \
            .TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self._tokens = set(self._tokenizer.tokens)
        self._indices = None
        if indices_path is not None:
            self._indices = np.loadtxt(indices_path, dtype=np.int32)

        self._cache = {}

    def __getitem__(self, index: int):
        if self._indices is not None:
            index = self._indices[index]
        res = self._cache.get(index)
        if res is not None:
            return res

        waveform, _, _, transcript = super().__getitem__(index)
        waveform = waveform.to(self._aligner.device).squeeze()

        transcript = transcript.lower()
        transcript = transcript.replace('"', "'")
        transcript = ''.join(filter(lambda c: c in self._tokens, transcript))

        waveform_length = torch.tensor([len(waveform)]).int() \
            .to(self._aligner.device)

        tokens, token_lengths = self._tokenizer(transcript)
        tokens = tokens.to(self._aligner.device).squeeze()
        token_lengths = token_lengths.to(self._aligner.device)
        assert token_lengths.item() == len(transcript), transcript

        durations = self._aligner(waveform[None, :], waveform_length,
                                  transcript).squeeze()

        res = (waveform, waveform_length, transcript,
               tokens, token_lengths, durations)
        self._cache[index] = res
        return res

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
