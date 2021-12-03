from functools import partial

from argparse_dataclass import ArgumentParser
import torch
from torchinfo import summary

from src.aligners import GraphemeAligner
from src.configs import FastSpeechConfig, FastSpeech2Config
from src.data_utils import collate, LJSpeechDataset, Wav2Spec
from src.models import FastSpeech
from src.trainers import DefaultTrainer
from src.utils import seed_all
from src.vocoders import WaveGlow


TRAIN_INDICES = 'data/lj_speech/train_indices.txt'
VAL_INDICES = 'data/lj_speech/val_indices.txt'


def lr_multiplier(config: FastSpeechConfig, step: int):
    step = step + 1
    return ((config.d_model ** -0.5) *
            min(step ** -0.5, step * (config.warmup_steps ** -1.5)))


seed_all()

config = ArgumentParser(FastSpeech2Config).parse_args()

fs = FastSpeech(config)
summary(fs)

aligner = GraphemeAligner(config)

train_ds = LJSpeechDataset(aligner, root=config.data_dir,
                           indices_path=TRAIN_INDICES)
val_ds = LJSpeechDataset(aligner, root=config.data_dir,
                         indices_path=VAL_INDICES)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=config.train_batch_size,
    shuffle=True,
    collate_fn=collate,
    num_workers=config.train_num_workers
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=config.val_batch_size,
    shuffle=True,
    collate_fn=collate,
    num_workers=config.val_num_workers
)

opt = torch.optim.Adam(
    fs.parameters(),
    lr=config.initial_lr,
    betas=config.betas,
    eps=config.eps,
    weight_decay=config.weight_decay
)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_multiplier,
                                                           config))
wav2spec = Wav2Spec(config)

vocoder = WaveGlow(config.device)

trainer = DefaultTrainer(
    config,
    fs,
    opt,
    scheduler,
    wav2spec,
    vocoder,
    train_loader=train_loader,
    val_loader=val_loader
)
trainer.train(num_epochs=config.num_epochs)
