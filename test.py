
from pathlib import Path

from argparse_dataclass import ArgumentParser
import torch
import torchaudio.pipelines as pipelines
from torchinfo import summary
import wandb

from src.aligners import GraphemeAligner
from src.configs import FastSpeech2Config
from src.data_utils import Wav2Spec
from src.models import FastSpeech
from src.utils import seed_all
from src.vocoders import WaveGlow


seed_all()

config = ArgumentParser(FastSpeech2Config).parse_args()

fs = FastSpeech(config)
summary(fs)

aligner = GraphemeAligner(config)

ds = [
    ('A defibrillator is a device that gives a high energy electric shock ' +
     'to the heart of someone who is in cardiac arrest.').lower(),
    ('Massachusetts Institute of Technology may be best known for its math, ' +
     'science and engineering education.').lower(),
    ('Wasserstein distance or Kantorovich Rubinstein metric ' +
     'is a distance function defined between probability distributions ' +
     'on a given metric space.').lower()
]

wav2spec = Wav2Spec(config)
vocoder = WaveGlow(config.device)
tokenizer = pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
device = config.device

wandb.init(job_type='test-model', config=config)
a = wandb.use_artifact('_username_/fast_speech/trainer_state:best',
                       type='trainer-state')
state_path = Path(a.download()) / 'state.pth'
state_dict = torch.load(state_path, map_location=config.device)
fs.load_state_dict(state_dict['model'])
fs.eval()

table = wandb.Table(columns=[
    'output_spec',
    'output_wav',
    'text'
])

for text in ds:
    tokens, _ = tokenizer(text)
    out_spec, _ = fs(tokens.to(device))
    out_wav = vocoder.inference(out_spec).squeeze().cpu()

    table.add_data(
        wandb.Image(out_spec.cpu()),
        wandb.Audio(out_wav, sample_rate=vocoder.OUT_SAMPLE_RATE),
        wandb.Html(text)
    )

wandb.log({'val results': table})
