from __future__ import annotations
import json
import re

LOSSES = {"mse", "l1", "smoothl1"}
SCOPES = {"head_only", "last_block_plus_head"}
TINY_FAMILIES = {"tiny_cnn", "feature_mlp"}


def _json(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("response contains no JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def validate_autoresearch_bundle(value: dict) -> dict:
    if set(value) != {"candidates"} or len(value["candidates"]) != 4:
        raise ValueError("exactly four candidates required")
    starts = ["population", "population", "fixed", "fixed"]
    clean = []
    for pos, row in enumerate(value["candidates"]):
        required = {
            "id",
            "start",
            "scope",
            "loss",
            "learning_rate",
            "weight_decay",
            "epochs",
            "batch_size",
            "patience",
            "hypothesis",
        }
        if set(row) != required or row["start"] != starts[pos]:
            raise ValueError("invalid candidate keys/order")
        if row["scope"] not in SCOPES or row["loss"] not in LOSSES:
            raise ValueError("invalid recipe")
        lr = float(row["learning_rate"])
        wd = float(row["weight_decay"])
        ep = int(row["epochs"])
        bs = int(row["batch_size"])
        patience = int(row["patience"])
        if (
            not 1e-6 <= lr <= 3e-3
            or not 0 <= wd <= 0.1
            or not 5 <= ep <= 40
            or bs not in {8, 16, 32}
            or not 2 <= patience <= 8
        ):
            raise ValueError("recipe out of bounds")
        clean.append(
            {
                **row,
                "id": f"ar_{pos + 1}",
                "learning_rate": lr,
                "weight_decay": wd,
                "epochs": ep,
                "batch_size": bs,
                "patience": patience,
                "hypothesis": str(row["hypothesis"])[:300],
            }
        )
    return {"candidates": clean}


def validate_tiny_bundle(value: dict) -> dict:
    if set(value) != {"candidates"} or len(value["candidates"]) != 4:
        raise ValueError("exactly four tiny candidates required")
    clean = []
    for pos, row in enumerate(value["candidates"]):
        required = {
            "id",
            "family",
            "width1",
            "width2",
            "kernel_size",
            "head_width",
            "dropout",
            "learning_rate",
            "weight_decay",
            "epochs",
            "batch_size",
            "loss",
            "hypothesis",
        }
        if (
            set(row) != required
            or row["family"] not in TINY_FAMILIES
            or row["loss"] not in LOSSES
        ):
            raise ValueError("invalid tiny spec")
        values = {
            "width1": int(row["width1"]),
            "width2": int(row["width2"]),
            "kernel_size": int(row["kernel_size"]),
            "head_width": int(row["head_width"]),
            "epochs": int(row["epochs"]),
            "batch_size": int(row["batch_size"]),
        }
        dropout = float(row["dropout"])
        learning_rate = float(row["learning_rate"])
        weight_decay = float(row["weight_decay"])
        if (
            not 4 <= values["width1"] <= 16
            or not 8 <= values["width2"] <= 32
            or values["kernel_size"] not in {3, 5, 7, 9}
            or not 8 <= values["head_width"] <= 32
            or not 8 <= values["epochs"] <= 40
            or values["batch_size"] not in {8, 16, 32}
            or not 0 <= dropout <= 0.5
            or not 1e-5 <= learning_rate <= 3e-3
            or not 0 <= weight_decay <= 0.1
        ):
            raise ValueError("tiny spec out of bounds")
        clean.append(
            {
                **row,
                **values,
                "id": f"tiny_{pos + 1}",
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "hypothesis": str(row["hypothesis"])[:300],
            }
        )
    return {"candidates": clean}
