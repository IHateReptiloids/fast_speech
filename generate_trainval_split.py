from pathlib import Path

import numpy as np
import torch

from src.utils import seed_all


# LJSpeech has 13100 samples
TOTAL_SAMPLES = 13100
TRAIN_FRAC = 0.95

TRAIN_PATH = Path('data/lj_speech/train_indices.txt')
VAL_PATH = Path('data/lj_speech/val_indices.txt')


seed_all()
permutation = torch.randperm(TOTAL_SAMPLES).numpy()
train_size = int(TOTAL_SAMPLES * TRAIN_FRAC)
train_indices = np.sort(permutation[:train_size])
val_indices = np.sort(permutation[train_size:])

np.savetxt(TRAIN_PATH, train_indices, fmt='%d')
np.savetxt(VAL_PATH, val_indices, fmt='%d')
