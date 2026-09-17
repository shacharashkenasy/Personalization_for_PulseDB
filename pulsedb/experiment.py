"""Three supported experiment modes and a review-before-evaluation boundary."""

from __future__ import annotations
import csv
import json
import math
import platform
from pathlib import Path
import numpy as np
import torch
from .config import ROOT, read, atomic_json, stable_hash, sha256_file
from .data import PulseDBFile, RestrictedData, personalization_splits
from .population import load_population_checkpoint, seed_everything
from .modeling import RECIPES, STANDARD_RECIPE_ID, fit_recipe, predict, metrics
from .tiny_model import predict_tiny
from .llm import Agent
from .fitting import fit_bank, artifact_model
from .memory import build_memory
from .evidence import calibration_cards, memory_cards
from .routing import (
    ROUTES,
    SHAPE_KEYS,
    INSTRUCTION,
    gate_pass,
    validation_scores,
    minimum_route,
    nearest_examples,
    schema,
    validate_answer,
    contract_reminder,
    override_guard,
)


class ReviewRequired(Exception):
    """An intentional pause with no held-out evaluation performed."""


def threshold(config, target):
    full = config["full_pipeline"]
    return {
        "mae_cap": full["deterministic_threshold"][target],
        "noninferiority": full["noninferiority"][target],
        "inflation_max": full["inflation_max"][target],
    }


def subject_splits(mat, cohort, protocol):
    splits = {
        s.subject: s
        for s in personalization_splits(
            mat, "memory" if cohort == "development" else "calfree"
        )
    }
    names = protocol[
        "development_subjects" if cohort == "development" else "calfree_subjects"
    ]
    hashes = protocol[
        "train_split_hashes" if cohort == "development" else "calfree_split_hashes"
    ]
    for name in names:
        if name not in splits or splits[name].hash != hashes[name]:
            raise ValueError(f"Dataset ordering/split mismatch: {cohort}/{name}")
    if cohort == "calfree" and set(splits) != set(names):
        raise ValueError("Expected exactly the published 144 CalFree subjects")
    return [splits[name] for name in names]


def clean_numbers(value):
    if isinstance(value, dict):
        return {k: clean_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_numbers(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_route(config, out, mat, split, bundle, route):
    if route == "abstain":
        return {"metrics": None, "predictions": [], "indices": [], "covered": False}
    if route == "population":
        model = load_population_checkpoint(
            config["paths"]["checkpoint_path"],
            mat.target,
            config["experiment"]["device"],
        )[0]
    else:
        model = artifact_model(bundle["artifacts"][route], out, config, mat.target)
    function = predict_tiny if route == "patient_only" else predict
    predictions = function(
        model, mat, split.final_holdout, config["experiment"]["device"]
    )
    truth = [mat.label(i) for i in split.final_holdout]
    return {
        "metrics": clean_numbers(metrics(truth, predictions)),
        "predictions": predictions.tolist(),
        "indices": list(split.final_holdout),
        "covered": True,
    }


def make_packet(config, out, mat, split, bundle, experiences, development):
    facts = calibration_cards(
        mat, split, out / "evidence" / mat.target / (split.subject + ".json")
    )
    facts.update(memory_cards(bundle, experiences))
    facts.update({"candidate." + r: bundle["metrics"][r] for r in ROUTES})
    examples = nearest_examples(mat.target, bundle["descriptor"], development)
    if len(examples) != 4:
        raise ValueError(
            "The guard requires four distinct training development examples"
        )
    for i, row in enumerate(examples):
        facts["memory.development_example_" + str(i)] = row
    scores = validation_scores(
        bundle["metrics"], {"search_weight": config["full_pipeline"]["search_weight"]}
    )
    return {
        "subject": split.subject,
        "target": mat.target,
        "budget": 160,
        "hard_integrity_failure": bundle["integrity"]["hard_integrity_failure"],
        "current_evaluation_data_included": False,
        "facts": facts,
        "validation_default": {
            "route": minimum_route(scores),
            "ranking_scores": scores,
            "search_weight": config["full_pipeline"]["search_weight"],
            "calibration_origin": "Separate training development patients only.",
        },
    }


def choose_route(config, out, mat, split, bundle, experiences, development, agent):
    path = out / "decisions" / mat.target / (split.subject + ".json")
    if path.exists():
        saved = read(path)
        if saved["bundle_hash"] != stable_hash(bundle):
            raise RuntimeError("Candidate evidence changed after routing")
        return saved
    hard = bundle["integrity"]["hard_integrity_failure"]
    passed = not hard and gate_pass(bundle["metrics"], threshold(config, mat.target))
    packet = None
    if hard:
        route, source, reasons, answer = "abstain", "hard_integrity_failure", [], None
    elif passed:
        route, source, reasons, answer = "autoresearch", "deterministic_gate", [], None
    else:
        packet = make_packet(config, out, mat, split, bundle, experiences, development)
        request = {"packet": packet, "schema": schema(packet["facts"])}
        atomic_json(out / "requests" / mat.target / (split.subject + ".json"), request)
        prompt = (
            INSTRUCTION
            + contract_reminder(packet)
            + "\nRESPONSE_SCHEMA:\n"
            + json.dumps(request["schema"])
        )
        try:
            answer = agent.call(
                kind="failure_aware_routing",
                prompt=prompt,
                evidence=packet,
                validator=lambda a: validate_answer(a, request),
                routing=True,
            )
            proposed = answer["route"]
            source = "agent_" + answer["decision"]
        except RuntimeError as exc:
            answer = None
            proposed = packet["validation_default"]["route"]
            source = "llm_error_validation_fallback"
            agent.record(
                {
                    "kind": "failure_aware_routing",
                    "status": "controller_fallback",
                    "error": str(exc),
                }
            )
        route, reasons = override_guard(packet, proposed, config["full_pipeline"])
        if reasons:
            source = "evidence_guard_validation_fallback"
    decision = {
        "subject": split.subject,
        "target": mat.target,
        "route": route,
        "selection_source": source,
        "gate_passed": bool(passed),
        "hard_integrity_failure": hard,
        "guard_reasons": reasons,
        "answer": answer,
        "bundle_hash": stable_hash(bundle),
        "packet_hash": stable_hash(packet),
        "threshold": threshold(config, mat.target),
    }
    atomic_json(path, decision)
    return decision


def review_decisions(config, out, choices):
    """HITL reviews failed-gate cases. Never auto-approve; reject stale approvals."""
    if not config["full_pipeline"]["hitl"]:
        return choices
    requests = []
    reviewed = []
    for choice in choices:
        if choice["gate_passed"]:
            reviewed.append(choice)
            continue
        review_hash = stable_hash(choice)
        path = out / "reviews" / choice["target"] / (choice["subject"] + ".json")
        if not path.exists():
            atomic_json(
                path,
                {
                    "status": "pending",
                    "decision_hash": review_hash,
                    "subject": choice["subject"],
                    "target": choice["target"],
                    "proposed_route": choice["route"],
                    "route": choice["route"],
                    "reviewer": "",
                    "rationale": "",
                },
            )
        review = read(path)
        if (
            review.get("decision_hash") != review_hash
            or review.get("subject") != choice["subject"]
            or review.get("target") != choice["target"]
        ):
            raise ValueError(f"Stale or mismatched human review: {path}")
        if review.get("status") != "approved":
            requests.append(str(path))
            continue
        if (
            not str(review.get("reviewer", "")).strip()
            or not str(review.get("rationale", "")).strip()
        ):
            raise ValueError(f"Approved review needs reviewer and rationale: {path}")
        allowed = (
            ["abstain"] if choice["hard_integrity_failure"] else ROUTES + ["abstain"]
        )
        if review.get("route") not in allowed:
            raise ValueError(
                f"Human review selected an unavailable/unsafe route: {path}"
            )
        reviewed.append(
            {
                **choice,
                "route": review["route"],
                "selection_source": "human_review",
                "human_review": review,
            }
        )
    if requests:
        atomic_json(
            out / "status.json",
            {
                "status": "awaiting_human_review",
                "pending": requests,
                "heldout_evaluation_started": False,
            },
        )
        raise ReviewRequired(
            f"{len(requests)} case(s) await review in {out / 'reviews'}. Edit status/route/reviewer/rationale and rerun the same command."
        )
    return reviewed


def prepare_development(config, protocol, out, experiences, agent):
    path = out / "development.json"
    if path.exists():
        return read(path)
    rows = []
    for target in config["experiment"]["targets"]:
        with PulseDBFile(
            Path(config["paths"]["data_path"]) / "VitalDB_Train_Subset.mat", target
        ) as mat:
            for pos, split in enumerate(subject_splits(mat, "development", protocol)):
                bundle = fit_bank(
                    config, out, mat, split, "development", experiences, agent
                )
                if bundle["integrity"]["hard_integrity_failure"]:
                    raise RuntimeError(
                        "Invalid Train development data: cannot create router examples"
                    )
                results = {
                    r: evaluate_route(config, out, mat, split, bundle, r)["metrics"][
                        "MAE"
                    ]
                    for r in ROUTES
                }
                rows.append(
                    {
                        "subject": split.subject,
                        "target": target,
                        "shape": {k: bundle["descriptor"][k] for k in SHAPE_KEYS},
                        "metrics": bundle["metrics"],
                        "internal_holdout_MAE": results,
                    }
                )
                print(f"Development {target}: {pos + 1}/48", flush=True)
    development = {
        "rows": rows,
        "source": "Regenerated Train internal holdout; no CalFree labels used",
    }
    atomic_json(path, development)
    return development


def initialize(config, protocol, out):
    required = ["VitalDB_CalFree_Test_Subset.mat"]
    if config["experiment"]["mode"] == "full_pipeline":
        required.append("VitalDB_Train_Subset.mat")
    files = {}
    for name in required:
        path = Path(config["paths"]["data_path"]) / name
        stat = path.stat()
        files[name] = {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    hashes = {
        t: sha256_file(
            Path(config["paths"]["checkpoint_path"]) / (t.lower() + "_resnet18_1d.pt")
        )
        for t in config["experiment"]["targets"]
    }
    source = {
        str(p.relative_to(ROOT)): sha256_file(p)
        for p in sorted((ROOT / "pulsedb").glob("*.py"))
    }
    identity = {
        "config": config,
        "protocol_hash": stable_hash(protocol),
        "datasets": files,
        "checkpoint_hashes": hashes,
        "source_hashes": source,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    manifest = out / "run_manifest.json"
    if manifest.exists():
        if read(manifest) != identity:
            raise ValueError(
                "Configuration, code, runtime, or inputs changed. Use a NEW output_dir."
            )
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Existing output directory has no matching run manifest")
        atomic_json(manifest, identity)
    return identity


def aggregate(out, rows, config):
    summary = {
        "mode": config["experiment"]["mode"],
        "calibration_type": config["calibration"]["type"],
        "budget": 160,
        "partial_cohort": config["experiment"]["limit"] is not None,
        "aggregation": "Unweighted mean of patient-level metrics; R2 is not pooled",
        "targets": {},
    }
    for target in config["experiment"]["targets"]:
        group = [r for r in rows if r["target"] == target]
        covered = [r for r in group if r["covered"]]
        values = {}
        for key in ("MAE", "RMSE", "R2", "bias"):
            finite = [
                r["metrics"][key] for r in covered if r["metrics"][key] is not None
            ]
            values[key] = float(np.mean(finite)) if finite else None
        summary["targets"][target] = {
            "patients": len(group),
            "covered_patients": len(covered),
            "coverage": len(covered) / len(group) if group else 0,
            **values,
        }
    atomic_json(out / "summary.json", summary)
    with (out / "per_patient.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "target",
                "route",
                "covered",
                "MAE",
                "RMSE",
                "R2",
                "bias",
                "n",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: row[k] for k in ["subject", "target", "route", "covered"]}
                | (row["metrics"] or {})
            )
    atomic_json(
        out / "status.json", {"status": "complete", "heldout_evaluation_started": True}
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run(config):
    protocol = read(ROOT / "configs/protocol.json")
    out = Path(config["paths"]["output_dir"])
    initialize(config, protocol, out)
    torch.set_num_threads(config["experiment"]["threads"])
    seed_everything(config["experiment"]["seed"])
    # Preserve original CUDA defaults instead of silently enabling reduced precision.
    device = config["experiment"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; set experiment.device=cpu")
    mode = config["experiment"]["mode"]
    if mode != "full_pipeline":
        rows = []
        for target in config["experiment"]["targets"]:
            base = load_population_checkpoint(
                config["paths"]["checkpoint_path"], target, device
            )[0]
            with PulseDBFile(
                Path(config["paths"]["data_path"]) / "VitalDB_CalFree_Test_Subset.mat",
                target,
            ) as mat:
                splits = subject_splits(mat, "calfree", protocol)[
                    : config["experiment"]["limit"]
                ]
                for pos, split in enumerate(splits):
                    destination = out / "results" / target / (split.subject + ".json")
                    if destination.exists():
                        rows.append(read(destination))
                        continue
                    calibrated = (
                        mode == "calibration"
                        and config["calibration"]["type"] == "finetune"
                    )
                    model = base
                    if calibrated:
                        view = RestrictedData(
                            mat, split.acquisition_pool + split.search_validation
                        )
                        seed = int(stable_hash([split.subject, target, 160])[:8], 16)
                        if config["experiment"]["seed"] != 1337:
                            seed = (seed + config["experiment"]["seed"] - 1337) % 2**32
                        model, _, _ = fit_recipe(
                            base,
                            view,
                            split.acquisition_pool,
                            split.search_validation,
                            RECIPES[STANDARD_RECIPE_ID],
                            device,
                            seed,
                        )
                    predictions = predict(model, mat, split.final_holdout, device)
                    row = {
                        "subject": split.subject,
                        "target": target,
                        "route": "fixed_calibration" if calibrated else "population",
                        "covered": True,
                        "metrics": clean_numbers(
                            metrics(
                                [mat.label(i) for i in split.final_holdout], predictions
                            )
                        ),
                        "predictions": predictions.tolist(),
                        "indices": list(split.final_holdout),
                        "split_hash": split.hash,
                    }
                    atomic_json(destination, row)
                    rows.append(row)
                    print(f"{mode} {target}: {pos + 1}/{len(splits)}", flush=True)
            del base
        return aggregate(out, rows, config)
    # Full pipeline always rebuilds non-backbone components in its own output directory.
    agent = Agent(
        config["llm"], out / "agent_calls.jsonl", config["experiment"]["seed"]
    )
    agent.check_ready()
    experiences = build_memory(config, protocol, out)
    development = prepare_development(config, protocol, out, experiences, agent)
    choices = []
    for target in config["experiment"]["targets"]:
        with PulseDBFile(
            Path(config["paths"]["data_path"]) / "VitalDB_CalFree_Test_Subset.mat",
            target,
        ) as mat:
            splits = subject_splits(mat, "calfree", protocol)[
                : config["experiment"]["limit"]
            ]
            for pos, split in enumerate(splits):
                bundle = fit_bank(
                    config, out, mat, split, "calfree", experiences, agent
                )
                choices.append(
                    choose_route(
                        config, out, mat, split, bundle, experiences, development, agent
                    )
                )
                print(
                    f"Full pipeline {target}: routed {pos + 1}/{len(splits)}",
                    flush=True,
                )
    choices = review_decisions(config, out, choices)
    # Commit ALL decisions (including real human reviews) before any CalFree scoring.
    frozen = out / "frozen_decisions.json"
    if frozen.exists() and read(frozen)["choices"] != choices:
        raise RuntimeError(
            "Refusing to change decisions after held-out evaluation was enabled"
        )
    atomic_json(frozen, {"choices": choices, "decisions_hash": stable_hash(choices)})
    lookup = {(c["target"], c["subject"]): c for c in choices}
    rows = []
    for target in config["experiment"]["targets"]:
        with PulseDBFile(
            Path(config["paths"]["data_path"]) / "VitalDB_CalFree_Test_Subset.mat",
            target,
        ) as mat:
            for split in subject_splits(mat, "calfree", protocol)[
                : config["experiment"]["limit"]
            ]:
                choice = lookup[target, split.subject]
                bundle = read(
                    out / "candidates/calfree" / target / split.subject / "bundle.json"
                )
                row = {
                    "subject": split.subject,
                    "target": target,
                    "route": choice["route"],
                    "decision_hash": stable_hash(choice),
                    **evaluate_route(config, out, mat, split, bundle, choice["route"]),
                }
                atomic_json(out / "results" / target / (split.subject + ".json"), row)
                rows.append(row)
    return aggregate(out, rows, config)
