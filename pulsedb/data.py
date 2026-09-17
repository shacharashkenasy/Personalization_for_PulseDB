from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
from .config import stable_hash


def _decode_reference(file: h5py.File, ref) -> str:
    chars = np.asarray(file[ref]).reshape(-1)
    return "".join(chr(int(value)) for value in chars)


class PulseDBFile:
    """Read-only MATLAB-v7.3 PulseDB subset with explicit ECG/PPG semantics."""

    def __init__(self, path: Path, target: str = "SBP"):
        self.path = Path(path).resolve()
        self.target = target.upper()
        if self.target not in {"SBP", "DBP"}:
            raise ValueError("target must be SBP or DBP")
        self.file = h5py.File(self.path, "r")
        self.subset = self.file["Subset"]
        self.n = int(self.subset[self.target].shape[-1])
        self._references: dict[str, np.ndarray] = {}
        self._text_cache: dict[tuple[str, int], str] = {}

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def _text(self, field: str, idx: int) -> str:
        if field not in self._references:
            self._references[field] = self.subset[field][0, :]
        ref = self._references[field][int(idx)]
        key = (field, hash(ref))
        if key not in self._text_cache:
            self._text_cache[key] = _decode_reference(self.file, ref)
        return self._text_cache[key]

    def subject(self, idx: int) -> str:
        return self._text("Subject", idx)

    def gender(self, idx: int) -> str:
        return self._text("Gender", idx)

    def signal(self, idx: int) -> np.ndarray:
        # Official PulseDB order: channel 0 ECG, channel 1 PPG, channel 2 ABP.
        value = np.asarray(self.subset["Signals"][:, 0:2, int(idx)], dtype=np.float32).T
        if value.shape[0] != 2:
            raise RuntimeError(f"expected ECG/PPG signal [2,T], found {value.shape}")
        return value

    def label(self, idx: int, target: str | None = None) -> float:
        target = (target or self.target).upper()
        return float(self.subset[target][0, int(idx)])

    def demographics(self, idx: int) -> dict[str, float]:
        idx = int(idx)
        result = {
            "age": float(self.subset["Age"][0, idx]),
            "bmi": float(self.subset["BMI"][0, idx]),
            "height": float(self.subset["Height"][0, idx]),
            "weight": float(self.subset["Weight"][0, idx]),
            "gender_male": float(self.gender(idx).strip().lower().startswith("m")),
        }
        return {
            key: (value if math.isfinite(value) else 0.0)
            for key, value in result.items()
        }

    def groups(self, probe_step: int = 40) -> tuple[list[str], dict[str, list[int]]]:
        """Discover exact contiguous subject blocks without decoding every string."""
        order: list[str] = []
        groups: dict[str, list[int]] = {}
        start = 0
        step = max(1, int(probe_step))
        while start < self.n:
            subject = self.subject(start)
            last_same = start
            probe = min(self.n - 1, start + step)
            while probe < self.n and self.subject(probe) == subject:
                last_same = probe
                if probe == self.n - 1:
                    break
                probe = min(self.n - 1, probe + step)
            if probe == self.n - 1 and self.subject(probe) == subject:
                stop = self.n
            else:
                low, high = last_same + 1, probe
                while low < high:
                    middle = (low + high) // 2
                    if self.subject(middle) == subject:
                        low = middle + 1
                    else:
                        high = middle
                stop = low
            if subject in groups:
                raise RuntimeError(f"non-contiguous subject encountered: {subject}")
            order.append(subject)
            groups[subject] = list(range(start, stop))
            start = stop
        return order, groups


@dataclass(frozen=True)
class PersonalizationSplit:
    subject: str
    acquisition_pool: tuple[int, ...]
    search_validation: tuple[int, ...]
    validator: tuple[int, ...]
    final_holdout: tuple[int, ...]
    holdout_role: str

    @property
    def hash(self) -> str:
        return stable_hash(
            {
                "subject": self.subject,
                "acquisition_pool": self.acquisition_pool,
                "search_validation": self.search_validation,
                "validator": self.validator,
                "final_holdout": self.final_holdout,
                "holdout_role": self.holdout_role,
            }
        )


def personalization_splits(mat: PulseDBFile, cohort: str) -> list[PersonalizationSplit]:
    order, groups = mat.groups()
    result = []
    final_n = 140 if cohort == "memory" else 180
    expected = 160 + 40 + 20 + final_n
    for subject in order:
        indices = groups[subject]
        if len(indices) != expected:
            raise ValueError(
                f"{cohort}/{subject}: expected {expected} windows, found {len(indices)}"
            )
        result.append(
            PersonalizationSplit(
                subject=subject,
                acquisition_pool=tuple(indices[:160]),
                search_validation=tuple(indices[160:200]),
                validator=tuple(indices[200:220]),
                final_holdout=tuple(indices[220:]),
                holdout_role="internal_holdout"
                if cohort == "memory"
                else "locked_test",
            )
        )
    return result


class RestrictedData:
    """Prevent fitting/agent evidence code from reading held-out signals or labels."""

    def __init__(self, mat, allowed):
        self._mat = mat
        self.allowed = frozenset(map(int, allowed))
        self.target = mat.target
        # At most 220 small windows per patient. Avoid repeated HDF5 reads on
        # every epoch/candidate without changing any input values or partitions.
        self._signals = {}
        self._labels = {}

    def _check(self, idx):
        if int(idx) not in self.allowed:
            raise PermissionError(
                f"Window {idx} is outside the fitting/validation partition"
            )

    def signal(self, idx):
        self._check(idx)
        idx = int(idx)
        if idx not in self._signals:
            value = self._mat.signal(idx)
            value.setflags(write=False)
            self._signals[idx] = value
        return self._signals[idx].copy(order="K")

    def label(self, idx, target=None):
        self._check(idx)
        key = (int(idx), target)
        if key not in self._labels:
            self._labels[key] = self._mat.label(idx, target)
        return self._labels[key]

    def demographics(self, idx):
        self._check(idx)
        return self._mat.demographics(idx)
