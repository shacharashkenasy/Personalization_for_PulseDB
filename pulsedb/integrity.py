from __future__ import annotations
import numpy as np
from .config import stable_hash


def audit_arrays(signals: np.ndarray, labels: np.ndarray | None = None) -> dict:
    signals = np.asarray(signals, dtype=np.float64)
    failures, warnings = [], []
    if signals.ndim != 3 or signals.shape[1] != 2:
        failures.append("invalid_ecg_ppg_shape")
        return {
            "passed": False,
            "hard_integrity_failure": True,
            "failures": failures,
            "warnings": warnings,
            "evidence_hash": stable_hash({"shape": signals.shape}),
        }
    finite = np.isfinite(signals)
    if not finite.all():
        failures.append("nonfinite_signal")
    safe = np.nan_to_num(signals)
    std = safe.std(axis=2)
    flat_fraction = np.mean(np.abs(np.diff(safe, axis=2)) < 1e-7, axis=2)
    bad_flat = (std < 1e-6) | (flat_fraction > 0.98)
    if float(np.mean(bad_flat)) > 0.05:
        failures.append("flatline_signal")
    extrema = np.mean(
        (safe == safe.min(axis=2, keepdims=True))
        | (safe == safe.max(axis=2, keepdims=True)),
        axis=2,
    )
    if np.any(extrema > 0.40):
        warnings.append("possible_clipping")
    if len(safe) > 1:
        fingerprints = [stable_hash(row.round(6).tolist()) for row in safe]
        if len(set(fingerprints)) < max(1, int(0.8 * len(fingerprints))):
            failures.append("excessive_duplicate_windows")
    # ECG normally has more high-frequency energy than PPG; this is a warning, never sole corruption proof.
    roughness = np.std(np.diff(safe, axis=2), axis=2).mean(axis=0)
    if roughness[0] < roughness[1] * 0.35:
        warnings.append("possible_channel_swap")
    correlations = [
        np.corrcoef(row[0], row[1])[0, 1] if row[0].std() and row[1].std() else 0
        for row in safe
    ]
    if np.nanmedian(np.abs(correlations)) < 0.01:
        warnings.append("possible_ecg_ppg_misalignment")
    best_lags = []
    for row in safe:
        left = row[0] - np.mean(row[0])
        right = row[1] - np.mean(row[1])
        corr = np.correlate(left, right, mode="full")
        best_lags.append(int(np.argmax(np.abs(corr)) - (len(left) - 1)))
    if np.median(np.abs(best_lags)) > signals.shape[-1] * 0.25:
        warnings.append("possible_ecg_ppg_misalignment")
    if labels is not None:
        labels = np.asarray(labels, dtype=np.float64)
        if len(labels) != len(signals) or not np.isfinite(labels).all():
            failures.append("invalid_labels")
        if len(labels) > 1 and np.ptp(labels) > 150:
            warnings.append("contradictory_label_range")
    failures = sorted(set(failures))
    warnings = sorted(set(warnings))
    evidence = {
        "shape": list(signals.shape),
        "finite_fraction": float(finite.mean()),
        "channel_std_median": np.median(std, axis=0).tolist(),
        "flat_fraction_max": float(flat_fraction.max()),
        "roughness": roughness.tolist(),
        "median_absolute_best_lag": float(np.median(np.abs(best_lags))),
        "failures": failures,
        "warnings": warnings,
    }
    return {
        "passed": not failures,
        "hard_integrity_failure": bool(failures),
        "failures": failures,
        "warnings": warnings,
        "summary": evidence,
        "evidence_hash": stable_hash(evidence),
    }
