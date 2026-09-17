"""The five internal models required by the full pipeline, at budget 160."""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .config import atomic_json, read, sha256_file, stable_hash
from .data import RestrictedData
from .population import load_population_checkpoint
from .modeling import (
    RECIPES,
    STANDARD_RECIPE_ID,
    fit_recipe,
    evaluate,
    metrics,
    predict,
)
from .tiny_model import fit_tiny, build_tiny, predict_tiny
from .features import feature_matrix
from .memory import current_descriptor, retrieve
from .integrity import audit_arrays
from .candidates import propose_bundles, recipe_from_candidate


@contextmanager
def time_limit(seconds):
    if seconds is None:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError(
            "Candidate timeouts require Linux/macOS; set candidate_timeout_seconds: null on Windows"
        )

    def expired(*_):
        raise TimeoutError(f"Candidate exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def artifact_model(artifact, out, config, target):
    path = Path(out) / artifact["path"]
    if sha256_file(path) != artifact["sha256"]:
        raise RuntimeError(f"Model artifact changed: {path}")
    payload = torch.load(
        path, map_location=config["experiment"]["device"], weights_only=False
    )
    if payload["kind"] == "tiny":
        model = build_tiny(payload["spec"]).to(config["experiment"]["device"])
    else:
        model = load_population_checkpoint(
            config["paths"]["checkpoint_path"], target, config["experiment"]["device"]
        )[0]
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def save_model(model, path, out, kind="resnet", spec=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {"kind": kind, "spec": spec, "state_dict": model.state_dict()}, temporary
    )
    temporary.replace(path)
    return {
        "path": str(path.relative_to(out)),
        "sha256": sha256_file(path),
        "kind": kind,
    }


def fit_bank(config, out, mat, split, cohort, experiences, agent):
    target, subject = mat.target, split.subject
    destination = out / "candidates" / cohort / target / subject
    bundle_path = destination / "bundle.json"
    if bundle_path.exists():
        bundle = read(bundle_path)
        if bundle["split_hash"] != split.hash:
            raise RuntimeError("Split changed on resume")
        for artifact in bundle["artifacts"].values():
            if sha256_file(out / artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("Candidate changed on resume")
        return bundle
    view = RestrictedData(
        mat, split.acquisition_pool + split.search_validation + split.validator
    )
    pool = list(split.acquisition_pool)
    base, _, checkpoint = load_population_checkpoint(
        config["paths"]["checkpoint_path"], target, config["experiment"]["device"]
    )
    checkpoint_hash = sha256_file(checkpoint)
    device = config["experiment"]["device"]
    labels = [view.label(i) for i in pool]
    integrity = audit_arrays(
        np.stack([view.signal(i) for i in pool]), np.asarray(labels)
    )
    if integrity["hard_integrity_failure"]:
        # Stop before fitting invalid data; caller emits a covered/not-covered result.
        bundle = {
            "subject": subject,
            "target": target,
            "split_hash": split.hash,
            "integrity": integrity,
            "artifacts": {},
            "metrics": {},
        }
        atomic_json(bundle_path, bundle)
        return bundle
    _, feature_rows = feature_matrix(view, pool)
    descriptor = current_descriptor(
        demographics=view.demographics(pool[0]),
        feature_rows=feature_rows,
        labels=labels,
    )
    proposals_path = destination / "proposals.json"
    if proposals_path.exists():
        proposals = read(proposals_path)
        ar, tiny = proposals["autoresearch"], proposals["patient_only"]
    else:
        ar, tiny = propose_bundles(
            agent,
            subject=subject,
            target=target,
            evidence={
                "budgets": [160],
                "integrity": integrity,
                "checkpoint_sha256": checkpoint_hash,
                "locked_test_available": False,
            },
        )
        atomic_json(proposals_path, {"autoresearch": ar, "patient_only": tiny})
    available = [
        r for r in experiences if r["target"] == target and r["subject"] != subject
    ]
    retrieval = retrieve(query=descriptor, experiences=available)
    by_id = {r["id"]: r for r in available}
    ranked = sorted(
        retrieval["records"],
        key=lambda r: (r["outcome"] != "success", r["distance"], r["id"]),
    )
    default_ids = [r["id"] for r in ranked[:3]]
    if len(default_ids) != 3:
        raise RuntimeError("At least three Train memory experiences required")
    allowed = {r["id"] for r in retrieval["records"]}

    def validate_memory(value):
        if set(value) != {"experience_ids", "rationale"}:
            raise ValueError("Expected experience_ids and rationale")
        ids = value["experience_ids"]
        if (
            len(ids) != 3
            or any(type(i) is not int for i in ids)
            or len(set(ids)) != 3
            or not set(ids) <= allowed
        ):
            raise ValueError("Select three distinct allowed Train experience IDs")
        return value

    memory_path = destination / "memory_selection.json"
    if memory_path.exists():
        choice = read(memory_path)
    else:
        try:
            choice = agent.call(
                kind="memory_route_selection",
                prompt="Select exactly three listed Train-only experience IDs. Use failures as avoidance evidence. Return JSON with experience_ids (three integers) and rationale (string).",
                evidence=[
                    {
                        k: v
                        for k, v in r.items()
                        if k
                        not in {
                            "adapter_path",
                            "adapter_sha256",
                            "distance_contributions",
                        }
                    }
                    for r in retrieval["records"]
                ],
                validator=validate_memory,
            )
        except RuntimeError as exc:
            choice = {
                "experience_ids": default_ids,
                "rationale": "Validation-compatible deterministic retrieval fallback",
                "fallback_error": str(exc),
            }
            agent.record(
                {
                    "kind": "memory_route_selection",
                    "status": "controller_fallback",
                    **choice,
                }
            )
        atomic_json(memory_path, choice)
    seed = int(stable_hash([subject, target, 160])[:8], 16)
    if config["experiment"]["seed"] != 1337:
        seed = (seed + config["experiment"]["seed"] - 1337) % 2**32
    timeout = config["full_pipeline"]["candidate_timeout_seconds"]
    all_trials = {}

    def search(name, specs, fit, offset):
        trials = []
        winner = None
        for pos, spec in enumerate(specs):
            start = time.monotonic()
            try:
                with time_limit(timeout):
                    model, score, epoch = fit(spec, (seed + offset + pos) % 2**32)
                key = (score["MAE"], spec["id"])
                trials.append(
                    {
                        "status": "complete",
                        "spec": spec,
                        "search": score,
                        "epoch": epoch,
                        "elapsed_seconds": time.monotonic() - start,
                    }
                )
                if winner is None or key < winner[0]:
                    winner = (key, model, spec, score)
            except (TimeoutError, RuntimeError, ValueError) as exc:
                trials.append({"status": "failed", "spec": spec, "error": str(exc)})
        all_trials[name] = trials
        if winner is None:
            raise RuntimeError(
                f"All {name} candidates failed; inspect logs or increase candidate_timeout_seconds"
            )
        return winner[1:]

    with time_limit(timeout):
        fixed, fixed_score, fixed_epoch = fit_recipe(
            base,
            view,
            pool,
            split.search_validation,
            RECIPES[STANDARD_RECIPE_ID],
            device,
            seed,
        )
    auto, auto_spec, auto_score = search(
        "autoresearch",
        ar["candidates"],
        lambda spec, s: fit_recipe(
            base if spec["start"] == "population" else fixed,
            view,
            pool,
            split.search_validation,
            recipe_from_candidate(spec),
            device,
            s,
        ),
        100,
    )
    mem_specs = []
    for experience_id in choice["experience_ids"]:
        row = by_id[experience_id]
        adapter = out / row["adapter_path"]
        if (
            row["checkpoint_sha256"] != checkpoint_hash
            or sha256_file(adapter) != row["adapter_sha256"]
        ):
            raise RuntimeError("Incompatible memory adapter")
        mem_specs.append(
            {
                "id": experience_id,
                "recipe_id": row["recipe_id"],
                "adapter": str(adapter),
                "warm_start": True,
            }
        )
    mem, mem_spec, mem_score = search(
        "memory",
        mem_specs,
        lambda spec, s: fit_recipe(
            base,
            view,
            pool,
            split.search_validation,
            RECIPES[spec["recipe_id"]],
            device,
            s,
            warm_start=Path(spec["adapter"]),
            target=target,
            checkpoint_sha256=checkpoint_hash,
        ),
        200,
    )
    patient, patient_spec, patient_score = search(
        "patient_only",
        tiny["candidates"],
        lambda spec, s: fit_tiny(
            mat=view,
            train_indices=pool,
            search_indices=split.search_validation,
            spec=spec,
            device=device,
            seed=s,
        ),
        300,
    )
    models = {
        "fixed_calibration": fixed,
        "autoresearch": auto,
        "memory": mem,
        "patient_only": patient,
    }
    scores = {
        "fixed_calibration": fixed_score,
        "autoresearch": auto_score,
        "memory": mem_score,
        "patient_only": patient_score,
    }
    measured = {
        "population": {
            "search_MAE": evaluate(base, view, split.search_validation, device)["MAE"],
            "validator_MAE": evaluate(base, view, split.validator, device)["MAE"],
        }
    }
    artifacts = {}
    for name, model in models.items():
        prediction = (
            predict_tiny(model, view, split.validator, device)
            if name == "patient_only"
            else predict(model, view, split.validator, device)
        )
        measured[name] = {
            "search_MAE": scores[name]["MAE"],
            "validator_MAE": metrics(
                [view.label(i) for i in split.validator], prediction
            )["MAE"],
        }
        artifacts[name] = save_model(
            model,
            destination / (name + ".pt"),
            out,
            kind="tiny" if name == "patient_only" else "resnet",
            spec=patient_spec if name == "patient_only" else None,
        )
    bundle = {
        "subject": subject,
        "target": target,
        "split_hash": split.hash,
        "integrity": integrity,
        "descriptor": descriptor,
        "retrieval": retrieval,
        "metrics": measured,
        "artifacts": artifacts,
        "memory_winner": {
            "experience_id": mem_spec["id"],
            "recipe_id": mem_spec["recipe_id"],
            "warm_start": mem_spec["warm_start"],
        },
        "trials": all_trials,
        "fit_indices": pool,
        "search_indices": list(split.search_validation),
        "routing_indices": list(split.validator),
    }
    atomic_json(bundle_path, bundle)
    return bundle
