import sys
from pathlib import Path

from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
import torch.nn as nn

sys.path.append('waveglow')
GDRIVE_FILE_ID = '1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF'


class WaveGlow(nn.Module):
    OUT_SAMPLE_RATE = 22050

    def __init__(
        self,
        device,
        checkpoint_path='data/waveglow/waveglow_256channels_universal_v5.pt',
    ):
        super().__init__()

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.exists():
            gdd.download_file_from_google_drive(
                file_id=GDRIVE_FILE_ID,
                dest_path=str(checkpoint_path)
            )
        model = torch.load(checkpoint_path, map_location=device)['model']
        self.net = model.remove_weightnorm(model)

    @torch.no_grad()
    def inference(self, spect: torch.Tensor):
        spect = self.net.upsample(spect)

        # trim the conv artifacts
        time_cutoff = self.net.upsample.kernel_size[0] - \
            self.net.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.net.n_group, self.net.n_group) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .flatten(start_dim=2) \
            .transpose(-1, -2)

        # generate prior
        audio = torch.randn(spect.size(0), self.net.n_remaining_channels,
                            spect.size(-1)).to(spect.device)

        for k in reversed(range(self.net.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.net.WN[k]((audio_0, spect))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.net.convinv[k](audio, reverse=True)

            if k % self.net.n_early_every == 0 and k > 0:
                z = torch.randn(
                    spect.size(0), self.net.n_early_size, spect.size(2),
                    device=spect.device
                )
                audio = torch.cat((z, audio), 1)

        audio = audio.permute(0, 2, 1) \
            .contiguous() \
            .view(audio.size(0), -1)

        return audio
