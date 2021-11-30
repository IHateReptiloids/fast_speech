import librosa
import torch
import torch.nn as nn
import torchaudio

from src.configs import FastSpeechConfig


class Wav2Spec(nn.Module):
    def __init__(self, config: FastSpeechConfig):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power
        )

        # Default `torchaudio` mel basis uses HTK formula.
        # In order to be compatible with WaveGlow
        # we decided to use Slaney one instead
        # (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T]
        """
        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel

    def transform_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(lengths, self.mel_spectrogram.hop_length,
                         rounding_mode='floor') + 1
