from __future__ import annotations
import math
from typing import Iterable
import numpy as np
from scipy.signal import find_peaks, peak_widths
from .data import PulseDBFile

WINDOW_FEATURE_KEYS = (
    "ecg_mean",
    "ecg_std",
    "ecg_range",
    "ecg_diff_std",
    "ecg_flat_fraction",
    "ppg_mean",
    "ppg_std",
    "ppg_range",
    "ppg_diff_std",
    "ppg_flat_fraction",
    "ecg_rate_bpm",
    "ecg_rr_cv",
    "ppg_rate_bpm",
    "ppg_pulse_width",
    "ecg_ppg_corr",
    "ecg_ppg_lag",
    "signal_quality",
    "temporal_position",
)


def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _rate_and_variability(
    values: np.ndarray, sample_rate: float = 125.0
) -> tuple[float, float, np.ndarray]:
    centered = values - np.median(values)
    distance = max(1, int(sample_rate * 0.30))
    prominence = max(float(np.std(centered)) * 0.35, 1e-6)
    peaks, _ = find_peaks(centered, distance=distance, prominence=prominence)
    if len(peaks) < 2:
        return 0.0, 0.0, peaks
    rr = np.diff(peaks) / sample_rate
    rate = 60.0 / max(float(np.median(rr)), 1e-6)
    cv = float(np.std(rr) / np.mean(rr)) if np.mean(rr) else 0.0
    return _finite(rate), _finite(cv), peaks


def window_features(
    signal: np.ndarray, temporal_position: float = 0.0
) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != 2:
        raise ValueError(f"signal must be [2,T] ECG/PPG, found {signal.shape}")
    ecg, ppg = signal
    result: dict[str, float] = {}
    for name, values in (("ecg", ecg), ("ppg", ppg)):
        diffs = np.diff(values)
        result.update(
            {
                f"{name}_mean": np.mean(values),
                f"{name}_std": np.std(values),
                f"{name}_range": np.ptp(values),
                f"{name}_diff_std": np.std(diffs),
                f"{name}_flat_fraction": np.mean(np.abs(diffs) < 1e-7),
            }
        )
    ecg_rate, ecg_cv, ecg_peaks = _rate_and_variability(ecg)
    ppg_rate, _, ppg_peaks = _rate_and_variability(ppg)
    width = 0.0
    if len(ppg_peaks):
        width = float(np.median(peak_widths(ppg, ppg_peaks, rel_height=0.5)[0]) / 125.0)
    corr = float(np.corrcoef(ecg, ppg)[0, 1]) if np.std(ecg) and np.std(ppg) else 0.0
    lag = 0.0
    if len(ecg_peaks) and len(ppg_peaks):
        delays = []
        for peak in ecg_peaks:
            future = ppg_peaks[ppg_peaks >= peak]
            if len(future) and future[0] - peak <= 125:
                delays.append((future[0] - peak) / 125.0)
        lag = float(np.median(delays)) if delays else 0.0
    quality = (
        float(np.isfinite(signal).mean())
        * max(0.0, 1.0 - result["ecg_flat_fraction"])
        * max(0.0, 1.0 - result["ppg_flat_fraction"])
    )
    result.update(
        {
            "ecg_rate_bpm": ecg_rate,
            "ecg_rr_cv": ecg_cv,
            "ppg_rate_bpm": ppg_rate,
            "ppg_pulse_width": width,
            "ecg_ppg_corr": corr,
            "ecg_ppg_lag": lag,
            "signal_quality": quality,
            "temporal_position": temporal_position,
        }
    )
    return {key: _finite(result[key]) for key in WINDOW_FEATURE_KEYS}


def feature_matrix(
    mat: PulseDBFile, indices: Iterable[int]
) -> tuple[np.ndarray, list[dict[str, float]]]:
    indices = list(map(int, indices))
    denominator = max(1, len(indices) - 1)
    rows = [
        window_features(mat.signal(idx), pos / denominator)
        for pos, idx in enumerate(indices)
    ]
    return np.asarray(
        [[row[key] for key in WINDOW_FEATURE_KEYS] for row in rows], dtype=np.float64
    ), rows
