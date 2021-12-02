from pathlib import Path
import string

import torch
import torchaudio


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root='data/lj_speech'):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        super().__init__(root=root, download=True)
        self._tokenizer = torchaudio.pipelines \
            .TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)

        to_remove = string.punctuation.replace("'", '')
        transcript = transcript.replace('"', "'")
        transcript = transcript.translate(str.maketrans('', '', to_remove))

        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)
        assert token_lengths.item() == len(transcript), transcript

        return waveform, waveform_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
