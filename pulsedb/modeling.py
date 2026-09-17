from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PulseDBFile
from .population import IndexedWindows


@dataclass(frozen=True)
class Recipe:
    id: str
    scope: str
    loss: str
    learning_rate: float
    epochs: int
    weight_decay: float
    batch_size: int
    patience: int


RECIPES = {
    item.id: item
    for item in (
        Recipe("head_mse_standard", "head_only", "mse", 1e-4, 20, 0.0, 16, 5),
        Recipe("head_l1_robust", "head_only", "l1", 1e-4, 25, 1e-5, 16, 6),
        Recipe("head_smoothl1", "head_only", "smoothl1", 3e-4, 25, 1e-5, 16, 6),
        Recipe(
            "lastblock_smoothl1",
            "last_block_plus_head",
            "smoothl1",
            3e-5,
            18,
            1e-5,
            8,
            5,
        ),
        Recipe(
            "lastblock_mse_low_lr", "last_block_plus_head", "mse", 1e-5, 15, 1e-5, 8, 5
        ),
    )
}
STANDARD_RECIPE_ID = "head_mse_standard"


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    errors = y_true - y_pred
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2)) if len(y_true) else 0.0
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "R2": float(1 - np.sum(errors**2) / denom) if denom else float("nan"),
        "bias": float(np.mean(errors)),
        "n": int(len(errors)),
    }


def predict(
    model,
    mat: PulseDBFile,
    indices: list[int] | tuple[int, ...],
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    loader = DataLoader(
        IndexedWindows(mat, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    values = []
    model.eval()
    with torch.no_grad():
        for signals, _labels, _indices in loader:
            values.extend(model(signals.to(device)).detach().cpu().numpy().reshape(-1))
    return np.asarray(values, dtype=np.float64)


def evaluate(
    model, mat: PulseDBFile, indices, device: str, batch_size: int = 128
) -> dict:
    truth = [mat.label(idx) for idx in indices]
    return metrics(truth, predict(model, mat, indices, device, batch_size))


def _configure(model, scope: str):
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = [model.fc] if scope == "head_only" else [model.layer4, model.fc]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def _set_train_mode(model, scope: str):
    model.eval()
    model.fc.train()
    if scope == "last_block_plus_head":
        model.layer4.train()


def fit_recipe(
    base_model,
    mat: PulseDBFile,
    train_indices,
    validation_indices,
    recipe: Recipe,
    device: str,
    seed: int,
    warm_start: Path | None = None,
    target: str | None = None,
    checkpoint_sha256: str | None = None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = copy.deepcopy(base_model).to(device)
    if warm_start:
        load_adapter(
            model, warm_start, target or mat.target, recipe.id, checkpoint_sha256
        )
    _configure(model, recipe.scope)
    loader = DataLoader(
        IndexedWindows(mat, list(train_indices)),
        batch_size=recipe.batch_size,
        shuffle=True,
        num_workers=0,
    )
    criterion = {
        "mse": torch.nn.MSELoss,
        "l1": torch.nn.L1Loss,
        "smoothl1": torch.nn.SmoothL1Loss,
    }[recipe.loss]()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        parameters, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    best_state, best_metrics, best_epoch, stale = (
        copy.deepcopy(model.state_dict()),
        None,
        0,
        0,
    )
    for epoch in range(1, recipe.epochs + 1):
        _set_train_mode(model, recipe.scope)
        for signals, labels, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(signals.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        current = evaluate(model, mat, validation_indices, device, recipe.batch_size)
        if best_metrics is None or current["MAE"] < best_metrics["MAE"]:
            best_metrics, best_epoch = current, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= recipe.patience:
            break
    model.load_state_dict(best_state)
    return model, best_metrics, best_epoch


def _adapter_state(model, scope: str) -> dict:
    prefixes = ("fc.",) if scope == "head_only" else ("layer4.", "fc.")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def save_adapter(
    path: Path,
    model,
    *,
    target: str,
    subject: str,
    recipe: Recipe,
    split_hash: str,
    checkpoint_sha256: str,
    metrics_by_split: dict,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 2,
            "target": target,
            "subject": subject,
            "recipe": asdict(recipe),
            "split_hash": split_hash,
            "checkpoint_sha256": checkpoint_sha256,
            "metrics": metrics_by_split,
            "adapter_state_dict": _adapter_state(model, recipe.scope),
        },
        temporary,
    )
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_adapter(
    model,
    path: Path,
    target: str,
    recipe_id: str,
    expected_checkpoint_sha256: str | None = None,
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["target"] != target or payload["recipe"]["id"] != recipe_id:
        raise ValueError("incompatible adapter target or recipe")
    base_hash = payload.get("checkpoint_sha256") or payload.get(
        "base_checkpoint_sha256"
    )
    if expected_checkpoint_sha256 and base_hash != expected_checkpoint_sha256:
        raise ValueError(
            "adapter was produced from an incompatible population checkpoint"
        )
    _missing, unexpected = model.load_state_dict(
        payload["adapter_state_dict"], strict=False
    )
    if unexpected:
        raise ValueError(f"unexpected adapter keys: {unexpected}")
    return payload
