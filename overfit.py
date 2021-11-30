from functools import partial
import pickle

import torch
from torchinfo import summary

from src.aligners import GraphemeAligner
from src.configs import FastSpeechConfig, FastSpeech2Config
from src.data_utils import collate, LJSpeechDataset, Wav2Spec
from src.models import FastSpeech
from src.trainers import DefaultTrainer
from src.utils import seed_all


BATCH_PATH = 'checkpoints/overfitted/batch.pkl'
MODEL_PATH = 'checkpoints/overfitted/model.pth'


def lr_multiplier(config: FastSpeechConfig, step: int):
    step = (step + 1) * 40
    return ((config.d_model ** -0.5) *
            min(step ** -0.5, step * (config.warmup_steps ** -1.5)))


seed_all()
config = FastSpeech2Config()
fs = FastSpeech(config)
summary(fs)

ds = LJSpeechDataset()
one_batch = []

loader = torch.utils.data.DataLoader(ds, batch_size=16,
                                     shuffle=True, collate_fn=collate)
for batch in loader:
    one_batch.append(batch)
    break

aligner = GraphemeAligner(config)
one_batch[0].durations = aligner(
    one_batch[0].waveform.to(config.device),
    one_batch[0].waveform_length.to(config.device),
    one_batch[0].transcript
)

opt = torch.optim.Adam(
    fs.parameters(),
    lr=1.0,
    betas=config.betas,
    eps=config.eps,
    weight_decay=config.weight_decay
)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_multiplier,
                                                           config))
wav2spec = Wav2Spec(config)

trainer = DefaultTrainer(
    config.device,
    fs,
    opt,
    scheduler,
    wav2spec,
    train_loader=one_batch,
    val_loader=None
)
trainer.train(num_epochs=config.num_epochs)
trainer.save_best_state('checkpoints/overfitted/model.pth')
with open(BATCH_PATH, 'wb') as f:
    pickle.dump(one_batch[0], f, pickle.HIGHEST_PROTOCOL)
