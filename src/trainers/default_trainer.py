from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data_utils import Batch


class DefaultTrainer:
    def __init__(
        self,
        device,
        model,
        opt,
        scheduler,
        wav2spec,
        train_loader,
        val_loader,
    ):
        self.device = device
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.wav2spec = wav2spec
        self.train_loader = train_loader
        self.val_loader = val_loader

        self._best_state = None
        self._best_loss = 1e9

    def save_best_state(self, path):
        path = Path(path)
        if path.exists():
            path.unlink()
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(self._best_state, path)

    def train(self, num_epochs, verbose=True):
        for i in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            if verbose:
                print(f'Epoch {i} train loss: {train_loss}')
            val_loss = self.validate()
            if val_loss < self._best_loss:
                self._best_loss = val_loss
                self._best_state = deepcopy(self.model.state_dict())
            if verbose:
                print(f'Epoch {i} validation loss: {val_loss}')
                print('-' * 100)

    def train_epoch(self):
        self._do_epoch(train=True)

    @torch.no_grad()
    def validate(self):
        self._do_epoch(train=False)

    def _do_epoch(self, train: bool):
        self.model.train(mode=train)
        total_loss = 0
        loader = self.train_loader if train else self.val_loader
        for batch in tqdm(loader):
            loss = self._process_batch(batch, train)
            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _process_batch(self, batch: Batch, train: bool):
        wavs = batch.waveform.to(self.device)
        assert wavs.dim() == 2

        wav_lengths = batch.waveform_length.to(self.device)
        assert wav_lengths.dim() == 1 and len(wav_lengths) == len(wavs)

        # specs is of shape [bs, n_mels, time]
        specs = self.wav2spec(wavs)
        spec_lengths = self.wav2spec.transform_lengths(wav_lengths)
        assert specs.shape[-1] == torch.max(spec_lengths).item()

        x = batch.tokens.to(self.device)
        output, predicted_lengths, output_lengths = None, None, None

        durations = batch.durations.to(self.device)
        assert durations.dim() == 2 and len(durations) == len(wavs)
        assert x.shape == durations.shape
        grapheme_lengths = (durations * spec_lengths[:, None])

        if train:
            output, predicted_lengths = self.model(x, grapheme_lengths)
            output_lengths = grapheme_lengths.round().int().sum(dim=-1)
        else:
            output, predicted_lengths = self.model(x)
            output_lengths = predicted_lengths.round().int().sum(dim=-1)

        assert (output.dim() == 3 and
                output.shape[-1] == torch.max(output_lengths).item())
        loss = 0
        for i in range(output.shape[0]):
            reshaped = F.interpolate(
                specs[i, :, :spec_lengths[i]].unsqueeze(0),
                size=output_lengths[i],
                mode='linear',
                align_corners=False
            )
            loss += F.mse_loss(output[i, :, :output_lengths[i]],
                               reshaped.squeeze())
        return loss / output.shape[0] + F.mse_loss(grapheme_lengths,
                                                   predicted_lengths)
