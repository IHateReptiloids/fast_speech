from collections import DefaultDict, OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.data_utils import Batch
from src.configs import FastSpeechConfig


@dataclass
class BatchLoss:
    loss: float
    spec_loss: float
    duration_loss: float
    padding_amount: float

    def __add__(self, other):
        loss = self.loss + other.loss
        spec_loss = self.spec_loss + other.spec_loss
        duration_loss = self.duration_loss + other.duration_loss
        padding_amount = self.padding_amount + other.padding_amount
        return BatchLoss(loss, spec_loss, duration_loss, padding_amount)

    def __truediv__(self, other: float):
        loss = self.loss / other
        spec_loss = self.spec_loss / other
        duration_loss = self.duration_loss / other
        padding_amount = self.padding_amount / other
        return BatchLoss(loss, spec_loss, duration_loss, padding_amount)


class DefaultTrainer:
    def __init__(
        self,
        config: FastSpeechConfig,
        model,
        opt,
        scheduler,
        wav2spec,
        vocoder,
        train_loader,
        val_loader,
    ):
        self.device = config.device
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.wav2spec = wav2spec
        self.vocoder = vocoder
        self.train_loader = train_loader
        self.val_loader = val_loader

        if config.checkpoint_path is not None:
            state = torch.load(config.checkpoint_path,
                               map_location=self.device)
            state['opt']['param_groups'][0]['initial_lr'] = config.initial_lr
            state['scheduler']['base_lrs'] = [config.initial_lr]
            self.load_state_dict(state)

        self.n_accumulate = config.n_accumulate
        self._accumulated = []

        self.train_log_freq = config.train_log_freq
        self.val_log_freq = config.val_log_freq
        self.checkpointing_freq = config.checkpointing_freq

        wandb.init(job_type='train-model', config=config)
        wandb.watch(model, log='parameters', log_freq=self.train_log_freq,
                    log_graph=True)
        self._checkpoint_path = Path(wandb.run.dir) / 'state.pth'

        self.best_state = OrderedDict()
        self.best_loss = 1e9

    def load_state_dict(self, d):
        self.model.load_state_dict(d['model'])
        self.opt.load_state_dict(d['opt'])
        self.scheduler.load_state_dict(d['scheduler'])

    def state_dict(self):
        state = OrderedDict()
        state['model'] = deepcopy(self.model.state_dict())
        state['opt'] = deepcopy(self.opt.state_dict())
        state['scheduler'] = deepcopy(
            self.scheduler.state_dict()
        )
        return state

    def train(self, num_epochs, verbose=True):
        for i in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            if verbose:
                print(f'Epoch {i} train loss: {train_loss}')
            val_loss = self.validate()
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._update_state()
            if verbose:
                print(f'Epoch {i} validation loss: {val_loss}')
                print('-' * 100)
            if i % self.checkpointing_freq == 0:
                p = (self._checkpoint_path.parent /
                     f'state{self.scheduler.last_epoch}.pth')
                torch.save(self.state_dict(), str(p))
        a = wandb.Artifact('trainer_state', type='trainer-state')
        a.add_file(self._checkpoint_path)
        wandb.log_artifact(a)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader):
            prepare_audio = (
                self.scheduler.last_epoch % self.train_log_freq == 0 and
                len(self._accumulated) + 1 == self.n_accumulate
            )
            indices = []
            bs = len(batch.transcripts)
            if prepare_audio:
                indices.append(torch.randint(0, bs, (1,)).item())
            loss, data = self._process_batch(batch, indices, train=True)
            self._accumulated.append(data['train/loss'])
            loss.backward()
            if len(self._accumulated) == self.n_accumulate:
                loss = (sum(self._accumulated, BatchLoss(0, 0, 0, 0)) /
                        self.n_accumulate)
                total_loss += loss.loss
                data.update({
                    'train/loss': loss.loss,
                    'train/spec_loss': loss.spec_loss,
                    'train/duration_loss': loss.duration_loss,
                    'train/padding_amount': loss.padding_amount,
                })
                wandb.log(data, step=self.scheduler.last_epoch)
                self._accumulated = []
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()
        return total_loss / (len(self.train_loader) / self.n_accumulate)

    @torch.no_grad()
    def validate(self, log_all=False):
        self.model.eval()
        total_loss = 0
        table = None
        for i, batch in enumerate(tqdm(self.val_loader)):
            prepare_audio = (i % self.val_log_freq == 0)
            indices = []
            bs = len(batch.transcripts)
            if prepare_audio:
                if log_all:
                    indices = list(range(bs))
                else:
                    indices.append(torch.randint(0, bs, (1,)).item())
            loss, data = self._process_batch(batch, indices, train=False)
            total_loss += loss.item()
            if prepare_audio:
                if table is None:
                    table = wandb.Table(columns=sorted(data.keys()))
                if log_all:
                    raise NotImplementedError
                data_ = list(zip(*sorted(data.items())))[1]
                table.add_data(*data_)

        total_loss /= len(self.val_loader)
        wandb.log({'val/audio': table, 'val/loss': total_loss},
                  step=self.scheduler.last_epoch)
        return total_loss

    def _prepare_audio(
        self,
        indices,
        gt_specs,
        gt_specs_lengths,
        out_specs,
        out_specs_lengths,
        transcripts,
        train: bool
    ):
        assert (len(gt_specs) == len(out_specs) == len(gt_specs_lengths) ==
                len(out_specs_lengths) == len(transcripts))
        res = DefaultDict(list)
        prefix = 'train/' if train else 'val/'
        for index in indices:
            gt_spec = gt_specs[index, :, :gt_specs_lengths[index]]
            out_spec = out_specs[index, :, :out_specs_lengths[index]]
            gt_wav = self.vocoder.inference(gt_spec.unsqueeze(0)).squeeze() \
                .cpu()
            out_wav = self.vocoder.inference(out_spec.unsqueeze(0)).squeeze() \
                .cpu()

            res[f'{prefix}ground_truth_spec'] \
                .append(wandb.Image(gt_spec.cpu()))
            res[f'{prefix}output_spec'].append(wandb.Image(out_spec.cpu()))
            res[f'{prefix}ground_truth_wav'].append(
                wandb.Audio(gt_wav,
                            sample_rate=self.vocoder.OUT_SAMPLE_RATE)
            )
            res[f'{prefix}output_wav'].append(
                wandb.Audio(out_wav,
                            sample_rate=self.vocoder.OUT_SAMPLE_RATE),
            )
            res[f'{prefix}text'].append(wandb.Html(transcripts[index]))
        if len(indices) == 1:
            for k in list(res.keys()):
                res[k] = res[k][0]
        return res

    def _process_batch(self, batch: Batch, indices, train: bool):
        '''
        indices are indices of audios in batch to pass to vocoder
        and return
        '''
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
        spec_loss = 0
        for i in range(output.shape[0]):
            reshaped = F.interpolate(
                specs[i, :, :spec_lengths[i]].unsqueeze(0),
                size=output_lengths[i],
                mode='linear',
                align_corners=False
            )
            spec_loss += F.mse_loss(output[i, :, :output_lengths[i]],
                                    reshaped.squeeze()) * output_lengths[i]
        spec_loss = spec_loss / (len(output) * output_lengths.sum())
        duration_loss = F.mse_loss(grapheme_lengths, predicted_lengths)
        loss = spec_loss + duration_loss

        padding_amount = (torch.max(output_lengths) - output_lengths) \
            .sum().item()

        data = {}
        if train:
            data = {'train/loss': BatchLoss(loss.item(), spec_loss.item(),
                                            duration_loss.item(),
                                            padding_amount),
                    'train/lr': self.scheduler.get_last_lr()[0]}
        data.update(self._prepare_audio(indices, specs, spec_lengths, output,
                                        output_lengths, batch.transcript,
                                        train))

        return loss, data

    def _update_state(self):
        wandb.summary['train/loss'] = self.best_loss
        self.best_state = deepcopy(self.state_dict())

        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()
        torch.save(self.best_state, str(self._checkpoint_path))
