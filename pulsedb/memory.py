from __future__ import annotations
import numpy as np
from .config import stable_hash

DISTANCE_KEYS = (
    "age",
    "bmi",
    "height",
    "weight",
    "gender_male",
    "ecg_mean",
    "ecg_std",
    "ecg_range",
    "ecg_diff_std",
    "ppg_mean",
    "ppg_std",
    "ppg_range",
    "ppg_diff_std",
    "calibration_target_mean",
    "calibration_target_std",
)


def current_descriptor(
    *, demographics: dict, feature_rows: list[dict], labels: list[float]
) -> dict:
    result = {
        key: float(demographics.get(key, 0.0))
        for key in ("age", "bmi", "height", "weight", "gender_male")
    }
    for key in (
        "ecg_mean",
        "ecg_std",
        "ecg_range",
        "ecg_diff_std",
        "ppg_mean",
        "ppg_std",
        "ppg_range",
        "ppg_diff_std",
    ):
        result[key] = float(np.mean([row[key] for row in feature_rows]))
    result["calibration_target_mean"] = float(np.mean(labels))
    result["calibration_target_std"] = float(np.std(labels))
    return result


def retrieve(
    *, query: dict, experiences: list[dict], success_k: int = 6, failure_k: int = 3
) -> dict:
    if not experiences:
        raise ValueError("memory is empty")
    matrix = np.asarray(
        [[row["descriptor"].get(k, 0.0) for k in DISTANCE_KEYS] for row in experiences]
    )
    q = np.asarray([query.get(k, 0.0) for k in DISTANCE_KEYS])
    scale = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
    scale[scale < 1e-8] = 1.0
    contributions = np.abs((matrix - q) / scale)
    distances = np.sqrt(np.mean(contributions**2, axis=1))
    ranked = sorted(
        range(len(experiences)), key=lambda i: (distances[i], int(experiences[i]["id"]))
    )
    selected = []
    for outcome, count in (("success", success_k), ("failure", failure_k)):
        candidates = [i for i in ranked if experiences[i]["outcome"] == outcome][:count]
        for i in candidates:
            row = {
                k: v
                for k, v in experiences[i].items()
                if k not in {"descriptor", "provenance_json"}
            }
            row["distance"] = float(distances[i])
            row["distance_contributions"] = {
                key: float(contributions[i, pos])
                for pos, key in enumerate(DISTANCE_KEYS)
            }
            selected.append(row)
    payload = {
        "records": selected,
        "query_hash": stable_hash(query),
        "source": "regenerated_train_only_memory192",
        "channel_contract_corrected": True,
    }
    payload["retrieval_hash"] = stable_hash(payload)
    return payload


def donor_descriptor(mat, indices):
    """Historical donor descriptor: global moments of the first 64 fitting windows.

    Correct channel names are used here; the old database called ECG 'ppg' and
    PPG 'abp'. Its numbers, not those obsolete names, define retrieval distances.
    """
    ids = list(indices)[:64]
    x = np.stack([mat.signal(i) for i in ids]).astype(np.float64)
    y = np.asarray([mat.label(i) for i in ids], dtype=np.float64)
    result = dict(mat.demographics(ids[0]))
    for channel, name in enumerate(("ecg", "ppg")):
        values = x[:, channel, :]
        result.update(
            {
                f"{name}_mean": float(values.mean()),
                f"{name}_std": float(values.std()),
                f"{name}_range": float(np.ptp(values, axis=1).mean()),
                f"{name}_diff_std": float(np.diff(values, axis=1).std()),
            }
        )
    result.update(
        calibration_target_mean=float(y.mean()), calibration_target_std=float(y.std())
    )
    return result


def build_memory(config, protocol, out):
    """Regenerate 192 Train donors, five recipes each, retaining two per target."""
    from pathlib import Path
    from .config import read, atomic_json, sha256_file
    from .data import PulseDBFile, RestrictedData
    from .population import load_population_checkpoint
    from .modeling import RECIPES, evaluate, fit_recipe, save_adapter

    experiences = []
    device = config["experiment"]["device"]
    for target in config["experiment"]["targets"]:
        base, _, checkpoint = load_population_checkpoint(
            config["paths"]["checkpoint_path"], target, device
        )
        checkpoint_hash = sha256_file(checkpoint)
        with PulseDBFile(
            Path(config["paths"]["data_path"]) / "VitalDB_Train_Subset.mat", target
        ) as mat:
            _, groups = mat.groups()
            for pos, subject in enumerate(protocol["memory_subjects"]):
                destination = out / "memory" / target / subject / "experiences.json"
                if destination.exists():
                    saved = read(destination)
                    for row in saved:
                        if (
                            sha256_file(out / row["adapter_path"])
                            != row["adapter_sha256"]
                        ):
                            raise RuntimeError("Regenerated memory adapter changed")
                    experiences.extend(saved)
                    continue
                indices = groups[subject]
                if len(indices) != 360:
                    raise ValueError("Train donor must contain 360 windows")
                # Donor-memory learning predates the 160/40/20/140 development split.
                fitting, validation, holdout = (
                    indices[:144],
                    indices[144:198],
                    indices[198:],
                )
                view = RestrictedData(mat, fitting + validation)
                descriptor = donor_descriptor(view, fitting)
                population = evaluate(base, view, validation, device, batch_size=64)
                trials = []
                for recipe in RECIPES.values():
                    seed = int(stable_hash([subject, target, recipe.id])[:8], 16)
                    if config["experiment"]["seed"] != 1337:
                        seed = (seed + config["experiment"]["seed"] - 1337) % 2**32
                    model, score, epoch = fit_recipe(
                        base, view, fitting, validation, recipe, device, seed
                    )
                    trials.append((score["MAE"], recipe, model, score, epoch))
                trials.sort(key=lambda item: (item[0], item[1].id))
                success = [t for t in trials if t[0] < population["MAE"]]
                selected = (success or trials)[:2]
                saved = []
                for rank, (_, recipe, model, score, epoch) in enumerate(selected):
                    test = evaluate(model, mat, holdout, device, batch_size=64)
                    adapter = destination.parent / (recipe.id + ".pt")
                    digest = save_adapter(
                        adapter,
                        model,
                        target=target,
                        subject=subject,
                        recipe=recipe,
                        split_hash=stable_hash([subject, fitting, validation, holdout]),
                        checkpoint_sha256=checkpoint_hash,
                        metrics_by_split={"validation": score, "holdout": test},
                    )
                    saved.append(
                        {
                            "id": pos * 2 + rank + 1,
                            "subject": subject,
                            "target": target,
                            "descriptor": descriptor,
                            "recipe_id": recipe.id,
                            "validation_mae": score["MAE"],
                            "population_validation_mae": population["MAE"],
                            "holdout_mae": test["MAE"],
                            "outcome": "failure"
                            if (test["MAE"] or score["MAE"]) >= population["MAE"]
                            else "success",
                            "adapter_path": str(adapter.relative_to(out)),
                            "adapter_sha256": digest,
                            "checkpoint_sha256": checkpoint_hash,
                            "best_epoch": epoch,
                        }
                    )
                atomic_json(destination, saved)
                experiences.extend(saved)
                del trials, selected, success, model
                print(
                    f"Memory {target}: {pos + 1}/{len(protocol['memory_subjects'])}",
                    flush=True,
                )
        del base
    return experiences
