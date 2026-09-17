# Validation performed on 2026-09-17

The standalone implementation was tested against the author's external PulseDB data and checkpoints. No data, trained models, recorded agent responses, result CSVs, or testing caches are included in this repository.

## Numerical checks

| Check | Patients per target | SBP mean MAE | DBP mean MAE | Difference from original |
|---|---:|---:|---:|---|
| Backbone: fresh held-out inference | 144 | 12.647189391 | 8.532576755 | Zero for every patient's MAE, RMSE, R² and bias |
| Population calibration: isolated standalone run | 144 | 12.647189391 | 8.532576755 | Identical to backbone |
| Fine-tune calibration: fresh training from backbone | 144 | 8.850071183 | 5.345843247 | Zero for every patient's MAE, RMSE, R² and bias |
| Full pipeline: external candidate-checkpoint verification | 144 | 5.720845171 | 3.287238774 | All 288 route decisions and patient MAEs matched |

The last row is **checkpoint-based verification**, not a new full LLM/training experiment. The standalone gate and guard revalidated the original external agent requests/responses, recomputed the route decisions, loaded the existing external candidate checkpoints, and ran fresh inference on the actual held-out signals. It did not simply average previously saved result numbers. The separate original results were used only for the final equality comparison.

## Fresh component checks

- Regenerated memory for one Train donor for both targets: five recipes per target, retaining two adapters per target. No copied donor model was used in this regeneration check.
- Refit the complete candidate bank for two selected CalFree patient-target cases (one SBP, one DBP). Original proposal responses and donor checkpoints were read externally to isolate the implementation check from new LLM sampling. Population, fixed calibration and patient-only validation MAEs matched; the largest AutoResearch/memory validation-MAE difference was **0.026700 mmHg**. These two selected cases are a bounded code check, not a cohort performance estimate.
- Exercised the generic HTTP client against a local test server with recorded proposal responses, plus an explicitly synthetic valid routing deferral. This verifies JSON transport, schema handling and control flow; it does not test a newly queried production LLM's quality.
- Reloaded newly generated patient-only model files and evaluated them successfully.
- Verified the per-patient fitting-data cache preserves input values and the held-out access boundary. Cached and uncached bounded candidate refits produced the same validation measurements.

## Isolation and control-flow checks

A copy of this repository ran from `/tmp`, outside the research checkout. A Python audit hook rejected attempts to open original-project files except the CalFree dataset and the two specified backbone checkpoints. The run completed for both targets and all 144 patients. Imported modules were checked: none came from the original research project.

**10 self-contained tests passed**, including MATLAB channel order and partition boundaries, held-out access rejection, all three gate conditions, guarded overrides, exact numeric evidence validation, generic HTTP retry/audit behavior, patient-weighted aggregation, HITL pause/resume, refusal to change approved routes after evaluation, and fitting-cache immutability. The full-mode controller's test replaces expensive fitting with fixtures but uses the real routing/review/freeze/evaluation ordering. Nearest-Train-example construction also matched all 73 original failed-gate request packets. Static undefined-name/unused-import checks passed.

Tested software: Python 3.12, PyTorch 2.10.0+cu128, NumPy 2.3.2, h5py 3.16.0, SciPy 1.16.1, PyYAML 6.0.2; NVIDIA L40S GPU. Runtime dependencies are pinned in `requirements.txt`; no package from the original project's dependency directories was needed.

## Remaining scope

A fresh, full-cohort run rebuilding all 192 donors and 48 development patients and sampling new production LLM responses was **not** completed for this cleanup. Its final MAE may differ from the checkpoint-based result above. Arbitrary alternative LLM models/providers, changed thresholds, and actual human overrides have no validated performance guarantee. The README's main reference is the original guarded pipeline with HITL disabled.
