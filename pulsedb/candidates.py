from __future__ import annotations

import json

from .llm import Agent
from .validators import validate_autoresearch_bundle, validate_tiny_bundle
from .modeling import Recipe
from .tiny_model import TINY_TEMPLATE


DEFAULT_AR = {
    "candidates": [
        {
            "id": "ar_1",
            "start": "population",
            "scope": "head_only",
            "loss": "smoothl1",
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "epochs": 25,
            "batch_size": 16,
            "patience": 5,
            "hypothesis": "robust head adaptation",
        },
        {
            "id": "ar_2",
            "start": "population",
            "scope": "last_block_plus_head",
            "loss": "l1",
            "learning_rate": 3e-5,
            "weight_decay": 1e-5,
            "epochs": 20,
            "batch_size": 8,
            "patience": 5,
            "hypothesis": "limited morphology adaptation",
        },
        {
            "id": "ar_3",
            "start": "fixed",
            "scope": "head_only",
            "loss": "mse",
            "learning_rate": 1e-4,
            "weight_decay": 0,
            "epochs": 20,
            "batch_size": 16,
            "patience": 5,
            "hypothesis": "refine fixed calibration",
        },
        {
            "id": "ar_4",
            "start": "fixed",
            "scope": "last_block_plus_head",
            "loss": "smoothl1",
            "learning_rate": 1e-5,
            "weight_decay": 1e-5,
            "epochs": 18,
            "batch_size": 8,
            "patience": 4,
            "hypothesis": "conservative fixed-start refinement",
        },
    ]
}
DEFAULT_TINY = {
    "candidates": [
        {**row, "id": f"tiny_{pos + 1}"} for pos, row in enumerate(TINY_TEMPLATE)
    ]
}


def propose_bundles(
    agent: Agent, *, subject: str, target: str, evidence: dict
) -> tuple[dict, dict]:
    ar_prompt = (
        "Modify numeric or categorical values in the JSON template below while preserving its structure exactly. "
        "Return one JSON object only: no markdown and no extra keys. Keep exactly four candidates. Keep their ids ar_1..ar_4 "
        "and starts in this exact order: population, population, fixed, fixed. The algorithms are fitted at "
        "160 windows. Use only supplied evidence; no locked test exists. TEMPLATE_JSON: "
        + json.dumps(DEFAULT_AR)
    )
    tiny_prompt = (
        "Modify numeric or categorical values in the JSON template below while preserving its structure exactly. "
        "Return one JSON object only: no markdown and no extra keys. Keep exactly four patient-only ECG+PPG models, ids "
        "tiny_1..tiny_4. All start from random weights, cannot use population or other-patient weights, and must remain "
        "under 100000 parameters. The specs are fitted at 160 windows. TEMPLATE_JSON: "
        + json.dumps(DEFAULT_TINY)
    )
    common = {"subject": subject, "target": target, **evidence}
    ar = agent.call(
        kind="autoresearch_bundle",
        prompt=ar_prompt,
        evidence=common,
        validator=validate_autoresearch_bundle,
    )
    tiny = agent.call(
        kind="patient_only_bundle",
        prompt=tiny_prompt,
        evidence=common,
        validator=validate_tiny_bundle,
    )
    return ar, tiny


def recipe_from_candidate(row: dict) -> Recipe:
    return Recipe(
        id=row["id"],
        scope=row["scope"],
        loss=row["loss"],
        learning_rate=float(row["learning_rate"]),
        epochs=int(row["epochs"]),
        weight_decay=float(row["weight_decay"]),
        batch_size=int(row["batch_size"]),
        patience=int(row["patience"]),
    )
