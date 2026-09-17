"""Configuration, artifact identity, and atomic JSON helpers."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value):
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def load_config(path, overrides=()):
    config = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                raise ValueError(f"Unknown config key: {key}")
            current = current[part]
        if parts[-1] not in current:
            raise ValueError(f"Unknown config key: {key}")
        current[parts[-1]] = yaml.safe_load(value)
    if config["experiment"]["mode"] not in {"full_pipeline", "backbone", "calibration"}:
        raise ValueError(
            "experiment.mode must be full_pipeline, backbone, or calibration"
        )
    if config["calibration"]["type"] not in {"population", "finetune"}:
        raise ValueError("calibration.type must be population or finetune")
    if type(config["full_pipeline"]["hitl"]) is not bool:
        raise ValueError("full_pipeline.hitl must be true or false")
    if not config["experiment"]["targets"] or not set(
        config["experiment"]["targets"]
    ) <= {"SBP", "DBP"}:
        raise ValueError("experiment.targets must contain SBP and/or DBP")
    if len(set(config["experiment"]["targets"])) != len(
        config["experiment"]["targets"]
    ):
        raise ValueError("experiment.targets must not contain duplicates")
    for key in ("seed", "threads"):
        value = config["experiment"][key]
        if type(value) is not int or value < (1 if key == "threads" else 0):
            raise ValueError(
                f"experiment.{key} must be a valid nonnegative integer (threads >= 1)"
            )
    if config["experiment"]["seed"] >= 2**32:
        raise ValueError("experiment.seed must be less than 2**32")
    if config["experiment"]["limit"] is not None and (
        type(config["experiment"]["limit"]) is not int
        or config["experiment"]["limit"] < 1
    ):
        raise ValueError("experiment.limit must be null or a positive integer")
    for key in ("data_path", "checkpoint_path", "output_dir"):
        expanded = os.path.expandvars(str(config["paths"][key]))
        if "$" in expanded:
            raise ValueError(f"Set paths.{key} or its environment variable: {expanded}")
        p = Path(expanded).expanduser()
        config["paths"][key] = str((p if p.is_absolute() else ROOT / p).resolve())
    for target in ("SBP", "DBP"):
        cap = config["full_pipeline"]["deterministic_threshold"][target]
        if (
            isinstance(cap, bool)
            or not isinstance(cap, (int, float))
            or not 0 < cap < float("inf")
        ):
            raise ValueError(
                "deterministic_threshold must specify positive finite SBP/DBP MAE caps"
            )
    if not 0 <= config["full_pipeline"]["search_weight"] <= 1:
        raise ValueError("search_weight must lie in [0,1]")
    for target in ("SBP", "DBP"):
        for key in ("noninferiority", "inflation_max"):
            value = config["full_pipeline"][key][target]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not 0 <= value < float("inf")
            ):
                raise ValueError(
                    f"full_pipeline.{key}.{target} must be finite and nonnegative"
                )
    timeout = config["full_pipeline"]["candidate_timeout_seconds"]
    if timeout is not None and (type(timeout) is not int or timeout < 1):
        raise ValueError("candidate_timeout_seconds must be null or a positive integer")
    support = config["full_pipeline"]["minimum_supporting_development_neighbors"]
    if type(support) is not int or not 0 <= support <= 4:
        raise ValueError(
            "minimum_supporting_development_neighbors must be an integer from 0 to 4"
        )
    regret = config["full_pipeline"]["maximum_relative_ranking_regret"]
    if not isinstance(regret, (float, int)) or not 0 <= regret < float("inf"):
        raise ValueError(
            "maximum_relative_ranking_regret must be finite and nonnegative"
        )
    for key in (
        "proposal_attempts",
        "routing_attempts",
        "max_tokens",
        "timeout_seconds",
    ):
        if type(config["llm"][key]) is not int or config["llm"][key] < 1:
            raise ValueError(f"llm.{key} must be a positive integer")
    return config
