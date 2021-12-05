
from pathlib import Path

from argparse_dataclass import ArgumentParser
import torch
from torchinfo import summary
import wandb

from src.aligners import GraphemeAligner
from src.configs import FastSpeech2Config
from src.data_utils import collate, LJSpeechDataset, Wav2Spec
from src.models import FastSpeech
from src.utils import seed_all
from src.vocoders import WaveGlow


VAL_INDICES = 'data/lj_speech/val_indices.txt'


seed_all()

config = ArgumentParser(FastSpeech2Config).parse_args()

fs = FastSpeech(config)
summary(fs)

aligner = GraphemeAligner(config)

val_ds = LJSpeechDataset(aligner, root=config.data_dir,
                         indices_path=VAL_INDICES)

val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=config.val_batch_size,
    shuffle=True,
    collate_fn=collate,
    num_workers=config.val_num_workers
)

wav2spec = Wav2Spec(config)
vocoder = WaveGlow(config.device)
device = config.device

wandb.init(job_type='test-model', config=config)
a = wandb.use_artifact('trainer_state:best')
state_path = Path(a.download()) / 'state.pth'
state_dict = torch.load(state_path, map_location=config.device)
fs.load_state_dict(state_dict['model'])

table = wandb.Table(columns=(
    'ground_truth_spec',
    'output_spec',
    'ground_truth_wav',
    'output_wav',
    'text'
))

for batch in val_loader:
    out_specs, out_lengths = fs(batch.tokens.to(device))
    out_lengths = out_lengths.round().int()

    gt_specs = wav2spec(batch.waveform.to(device))
    gt_lengths = wav2spec.transform_lengths(batch.waveform_length.to(device)) \
        .round().int()

    transcript = batch.transcript

    assert (len(out_specs) == len(out_lengths) ==
            len(gt_specs) == len(gt_lengths) == len(transcript))
    bs = len(out_specs)
    for index in range(bs):
        gt_spec = gt_specs[index, :, :gt_lengths[index]]
        out_spec = out_specs[index, :, :out_lengths[index]]
        gt_wav = vocoder.inference(gt_spec.unsqueeze(0)).squeeze().cpu()
        out_wav = vocoder.inference(out_spec.unsqueeze(0)).squeeze().cpu()

        table.add_data(
            wandb.Image(gt_spec.cpu()),
            wandb.Image(out_spec.cpu()),
            wandb.Audio(gt_wav, sample_rate=vocoder.OUT_SAMPLE_RATE),
            wandb.Audio(out_wav, sample_rate=vocoder.OUT_SAMPLE_RATE),
            wandb.Html(transcript[index])
        )
wandb.log({'val results': table})
