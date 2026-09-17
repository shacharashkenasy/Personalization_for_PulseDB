from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .modeling import metrics
from .population import IndexedWindows


class TinyPatientCNN(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        w1, w2 = int(spec["width1"]), int(spec["width2"])
        k = int(spec["kernel_size"])
        self.features = nn.Sequential(
            nn.Conv1d(2, w1, k, padding=k // 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(w1, w2, k, padding=k // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w2, int(spec["head_width"])),
            nn.ReLU(),
            nn.Dropout(float(spec["dropout"])),
            nn.Linear(int(spec["head_width"]), 1),
        )

    def forward(self, x):
        return self.head(self.features(x))


class PatientFeatureMLP(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        width = int(spec["head_width"])
        self.net = nn.Sequential(
            nn.Linear(12, width),
            nn.ReLU(),
            nn.Dropout(float(spec["dropout"])),
            nn.Linear(width, 1),
        )

    @staticmethod
    def summarize(x):
        mean = x.mean(-1)
        std = x.std(-1)
        minimum = x.amin(-1)
        maximum = x.amax(-1)
        diff = torch.diff(x, dim=-1).std(-1)
        energy = (x * x).mean(-1).sqrt()
        return torch.cat((mean, std, minimum, maximum, diff, energy), dim=1)

    def forward(self, x):
        return self.net(self.summarize(x))


def build_tiny(spec: dict) -> nn.Module:
    model = (
        TinyPatientCNN(spec)
        if spec["family"] == "tiny_cnn"
        else PatientFeatureMLP(spec)
    )
    count = sum(p.numel() for p in model.parameters())
    if count > 100_000:
        raise ValueError(f"patient-only model has {count} parameters")
    model.parameter_count = count
    return model


def predict_tiny(model, mat, indices, device, batch_size=128):
    loader = DataLoader(
        IndexedWindows(mat, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    output = []
    model.eval()
    with torch.no_grad():
        for signals, _labels, _idx in loader:
            output.extend(model(signals.to(device)).cpu().numpy().reshape(-1))
    return np.asarray(output, dtype=np.float64)


def fit_tiny(*, mat, train_indices, search_indices, spec, device, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_tiny(spec).to(device)
    loader = DataLoader(
        IndexedWindows(mat, list(train_indices)),
        batch_size=int(spec["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    criterion = {"mse": nn.MSELoss, "l1": nn.L1Loss, "smoothl1": nn.SmoothL1Loss}[
        spec["loss"]
    ]()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    truth = np.asarray([mat.label(i) for i in search_indices])
    best = None
    best_state = None
    stale = 0
    for epoch in range(1, int(spec["epochs"]) + 1):
        model.train()
        for signals, labels, _idx in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(signals.to(device)), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        current = metrics(truth, predict_tiny(model, mat, search_indices, device))
        if best is None or current["MAE"] < best["MAE"]:
            best = current
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best_state)
    return model, best, best_epoch


TINY_TEMPLATE = [
    {
        "id": "fixed_tiny_1",
        "family": "tiny_cnn",
        "width1": 8,
        "width2": 16,
        "kernel_size": 5,
        "head_width": 16,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "epochs": 25,
        "batch_size": 16,
        "loss": "smoothl1",
        "hypothesis": "compact morphology encoder",
    },
    {
        "id": "fixed_tiny_2",
        "family": "tiny_cnn",
        "width1": 12,
        "width2": 24,
        "kernel_size": 7,
        "head_width": 16,
        "dropout": 0.2,
        "learning_rate": 5e-4,
        "weight_decay": 1e-2,
        "epochs": 30,
        "batch_size": 16,
        "loss": "l1",
        "hypothesis": "wider robust morphology encoder",
    },
    {
        "id": "fixed_tiny_3",
        "family": "feature_mlp",
        "width1": 4,
        "width2": 8,
        "kernel_size": 3,
        "head_width": 32,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "epochs": 30,
        "batch_size": 16,
        "loss": "smoothl1",
        "hypothesis": "patient-only statistical feature model",
    },
    {
        "id": "fixed_tiny_4",
        "family": "feature_mlp",
        "width1": 4,
        "width2": 8,
        "kernel_size": 3,
        "head_width": 16,
        "dropout": 0,
        "learning_rate": 2e-3,
        "weight_decay": 0,
        "epochs": 25,
        "batch_size": 8,
        "loss": "mse",
        "hypothesis": "low-capacity patient-only baseline",
    },
]
