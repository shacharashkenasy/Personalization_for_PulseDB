from __future__ import annotations
from .config import atomic_json as write
from .features import window_features


class ValidationReader:
    """Enforce actual signal/label accesses; no held-out window is readable here."""

    def __init__(self, mat, allowed):
        self.mat, self.allowed = mat, set(map(int, allowed))
        self.signal_reads, self.label_reads = set(), set()

    def signal(self, idx):
        if int(idx) not in self.allowed:
            raise PermissionError("Signal outside fitting/validation partitions")
        self.signal_reads.add(int(idx))
        return self.mat.signal(idx)

    def label(self, idx):
        if int(idx) not in self.allowed:
            raise PermissionError("Label outside fitting/validation partitions")
        self.label_reads.add(int(idx))
        return self.mat.label(idx)


def distribution(values):
    import numpy as np

    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Nonfinite/empty feature distribution")
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "q10": float(np.quantile(x, 0.1)),
        "median": float(np.median(x)),
        "q90": float(np.quantile(x, 0.9)),
        "iqr": float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
    }


def shift(reference, current):
    result = {}
    for key, a in reference.items():
        b = current[key]
        # No arbitrary unit-valued scale for near-constant normalized features.
        denominator = a["iqr"]
        result[key] = {
            "mean_difference": b["mean"] - a["mean"],
            "reference_iqr": denominator,
            "mean_difference_in_reference_iqr": (b["mean"] - a["mean"]) / denominator
            if denominator > 1e-8
            else None,
        }
    return result


def calibration_cards(mat, split, destination):
    indices = {
        "fitting": list(split.acquisition_pool),
        "search_validation": list(split.search_validation),
        "routing_validation": list(split.validator),
    }
    reader = ValidationReader(mat, sum(indices.values(), []))
    rows, stats = {}, {}
    for name, ids in indices.items():
        rows[name] = []
        for idx in ids:
            features = window_features(reader.signal(idx))
            rows[name].append(
                {
                    "index": idx,
                    "label": reader.label(idx),
                    **{k: features[k] for k in FEATURE_KEYS},
                }
            )
        stats[name] = {
            k: distribution([r[k] for r in rows[name]])
            for k in ["label"] + FEATURE_KEYS
        }
    half = len(rows["fitting"]) // 2
    early = {
        k: distribution([r[k] for r in rows["fitting"][:half]])
        for k in ["label"] + FEATURE_KEYS
    }
    late = {
        k: distribution([r[k] for r in rows["fitting"][half:]])
        for k in ["label"] + FEATURE_KEYS
    }
    write(
        destination,
        {
            "rows": rows,
            "stats": stats,
            "early_fitting": early,
            "late_fitting": late,
            "allowed_indices": sorted(reader.allowed),
            "signal_reads": sorted(reader.signal_reads),
            "label_reads": sorted(reader.label_reads),
            "split_hash": split.hash,
        },
    )
    return {
        "calibration.fitting": stats["fitting"],
        "calibration.search_shift": shift(stats["fitting"], stats["search_validation"]),
        "calibration.routing_shift": shift(
            stats["fitting"], stats["routing_validation"]
        ),
        "calibration.early_late_fitting_shift": shift(early, late),
    }


SIGNAL_KEYS = [
    "ecg_mean",
    "ecg_std",
    "ecg_range",
    "ecg_diff_std",
    "ppg_mean",
    "ppg_std",
    "ppg_range",
    "ppg_diff_std",
]
DESCRIPTOR_KEYS = SIGNAL_KEYS + ["calibration_target_mean", "calibration_target_std"]
FEATURE_KEYS = [
    "ecg_std",
    "ecg_diff_std",
    "ppg_mean",
    "ppg_std",
    "ppg_diff_std",
    "ecg_rate_bpm",
    "ecg_rr_cv",
    "ppg_rate_bpm",
    "ppg_pulse_width",
    "ecg_ppg_corr",
    "ecg_ppg_lag",
    "signal_quality",
]


def memory_cards(bundle, experiences):
    target = bundle["target"]
    # Exclude the current development patient, including both of its recipes.
    records = [
        r
        for r in experiences
        if r["target"] == target and r["subject"] != bundle["subject"]
    ]
    profiles = {r["subject"]: r["descriptor"] for r in records}
    reference = {
        k: distribution([p[k] for p in profiles.values()]) for k in DESCRIPTOR_KEYS
    }
    query = {k: bundle["descriptor"][k] for k in DESCRIPTOR_KEYS}
    winner = bundle["memory_winner"]
    facts = {
        "memory.training_reference": {
            "unique_training_patients": len(profiles),
            "feature_distributions": reference,
        },
        "memory.current_calibration_descriptor": query,
        "memory.model_provenance": {
            "winner_experience_id": winner["experience_id"],
            "recipe_id": winner["recipe_id"],
            "donor_weights_actually_reused": winner["warm_start"],
            "otherwise_initialization": "population",
            "selection": "Search-validation winner.",
        },
    }
    by_id = {r["id"]: r for r in records}
    for saved in bundle["retrieval"]["records"]:
        row = by_id[saved["id"]]
        gaps = {
            k: {
                "current": query[k],
                "donor": row["descriptor"][k],
                "signed_difference_in_training_iqr": (query[k] - row["descriptor"][k])
                / reference[k]["iqr"]
                if reference[k]["iqr"] > 1e-8
                else None,
            }
            for k in DESCRIPTOR_KEYS
        }
        facts["memory.example_" + str(row["id"])] = {
            "train_subject": row["subject"],
            "recipe_id": row["recipe_id"],
            "donor_validation_MAE": row["validation_mae"],
            "donor_population_validation_MAE": row["population_validation_mae"],
            "donor_validation_MAE_reduction": row["population_validation_mae"]
            - row["validation_mae"],
            "original_retrieval_distance": saved["distance"],
            "descriptor_comparison": gaps,
            "used_as_memory_route_winner": row["id"] == winner["experience_id"],
        }
    return facts
