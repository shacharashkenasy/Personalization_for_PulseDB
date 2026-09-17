from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import PulseDBFile
from .resnet import Resnet18_1D


class IndexedWindows(Dataset):
    def __init__(self, mat: PulseDBFile, indices: list[int]):
        self.mat, self.indices = mat, list(map(int, indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        idx = self.indices[position]
        return (
            self.mat.signal(idx),
            np.asarray([self.mat.label(idx)], dtype=np.float32),
            idx,
        )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_population_checkpoint(checkpoint_dir, target, device):
    path = Path(checkpoint_dir) / f"{target.lower()}_resnet18_1d.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = Resnet18_1D().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload, path
