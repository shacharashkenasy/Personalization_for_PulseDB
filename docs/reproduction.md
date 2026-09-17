# Reproduction protocol and source trace

The numerical target comes from `guarded_agent_v1_qwen30b_20260913`, whose candidate models were trained in `routing_qwen30b_2026-09-04_v002`. The guarded run reused those models and changed route selection. Its exact cohort means were **5.7208451713949175 SBP / 3.287238774163165 DBP mmHg**. Ordinary rounding gives **5.72 / 3.29**; the requested “5.72 / 3.28” refers to this result.

These names identify provenance; no historical run directory is a runtime input to this repository.

## Preserved path

| Original component | Standalone component | Preserved behavior |
|---|---|---|
| `pulsedb_routing/data.py` | `pulsedb/data.py` | HDF5 access, ECG/PPG order, no extra normalization, contiguous patient splits |
| Legacy `Model_Def/ResNet.py` | `pulsedb/resnet.py` | Exact backbone architecture/state-dict layout |
| `pulsedb_routing/modeling.py` | `pulsedb/modeling.py` | Optimizer, losses, scopes, early stopping, batch sizes, prediction and metric formulas |
| `pulsedb_routing/candidates.py`, `agents.py`, `tiny_model.py` | `candidates.py`, `validators.py`, `tiny_model.py`, `llm.py` | Four bounded AutoResearch recipes and four compact patient-only proposals; provider-neutral transport |
| `pulsedb_memory/experiment.py`, `data.py`, `recipes.py` | `pulsedb/memory.py` | Regenerate donor experiences/adapters from Train data and backbone weights |
| `pulsedb_routing/memory.py`, `pipeline.py` | `memory.py`, `fitting.py` | Six successful/three unsuccessful retrieved experiences, three memory candidates, compatible donor warm starts |
| `run_pulsedb_deviation_router.py` evidence functions | `pulsedb/evidence.py` | Measured fitting/validation shifts, cohort/donor distributions and model provenance |
| `run_pulsedb_selective_router_v2.py` | `pulsedb/routing.py` | Five eligible internal routes, calibrated validation ranking, four nearest Train development examples, response validation |
| `run_pulsedb_guarded_router.py` | `pulsedb/routing.py` | Guarded overrides: no validation dominance, <=5% ranking regret, >=3/4 supporting peers |
| Original locked evaluation | `pulsedb/experiment.py` | Freeze all routes before CalFree evaluation; 180 held-out windows, mean patient metrics |

The production dependency graph is `run.py → pulsedb/* → external subsets + backbone weights + configured LLM`. Full-pipeline outputs are generated in `paths.output_dir`. The original project, its SQLite memory, model-server runtime, trained patient models, and recorded agent responses are not required or distributed.

## Data partitions

For each CalFree patient, in original file order:

- Windows 0–159: fitting (160).
- Windows 160–199: search validation (40), including candidate/epoch selection.
- Windows 200–219: routing validation (20).
- Windows 220–399: held-out evaluation (180).

The 48 development patients use the same first three partitions and a 140-window Train internal holdout. Their identities, the 192 memory donors, the 144 CalFree patients, and original split hashes are in `configs/protocol.json`. This is a split definition, without labels, predictions, or learned decisions. Train donors and CalFree patients are disjoint; development subjects belong to the Train donor pool. A development patient's own memory experiences are excluded when fitting its candidates.

Historical donor-memory training uses a **different** split of each Train patient's 360 windows: 144 fitting, 54 validation, and 162 internal holdout. It tries the five original recipes and stores the best two validation-improving recipes (or the best available recipes). Donor descriptors use global signal moments over the first 64 fitting windows. Their historical mislabeled channel names are corrected without changing channel order. Current-patient descriptors average per-window statistics over the 160 fitting windows, as in the reported pipeline.

Models receive only ECG/PPG signals. Calibration BP labels are used only in fitting and allowed validation/development evidence. Restricted readers block CalFree held-out access during model fitting and agent evidence construction. The original donor/development cohorts were not a new external validation set for the population backbone.

## Model search and routing

The fixed calibration recipe trains only the prediction head: Adam, MSE, learning rate `1e-4`, 20 epochs maximum, batch size 16, patience 5. AutoResearch proposes four candidates once per patient/target: two population starts and two fixed-calibration starts. Search ranges and validators are in `validators.py`; selection uses search-validation MAE, then candidate ID. There is no iterative feedback of evaluation outcomes to the LLM.

The memory route fits three retrieved recipes, with compatible donor adapters. The patient-only route selects among four randomly initialized models of at most 100,000 parameters. These are internal full-pipeline components, not separately supported comparison modes. Shuffled memory, fixed-tiny comparisons, HPO comparisons, other routers, and the 20/40-window budget sweeps are omitted.

The published gate constants are frozen defaults, not retuned against CalFree:

| Target | Routing-validation MAE cap | Allowed excess over fixed calibration | Routing/search MAE ratio cap |
|---|---:|---:|---:|
| SBP | 8.497991049265261 | 10% | 1.5 |
| DBP | 5.934352619117226 | 0% | 1.25 |

All three conditions must pass to retain AutoResearch automatically. Failure triggers the one guarded agent policy; AutoResearch remains eligible. Default ranking is `0.75 × search MAE + 0.25 × routing MAE`. That weight was selected on the original 48 Train development patients and is retained as a fixed configuration default. The original all-budget gate search is not rerun because this repository supports the main 160-window experiment only. Train development outcomes are regenerated for the four-neighbor evidence/override guard.

The router prompt and exact response schema are in `routing.py`. Unsupported or invalid overrides defer to the validation default. An unavailable routing LLM also causes an explicitly logged fallback, not a fabricated agent response. No silently substituted AutoResearch proposals are used.

## Checkpoint identities

| File | SHA-256 |
|---|---|
| `sbp_resnet18_1d.pt` | `d3f0ac7bef8db592b1e588a16ab4dba478ef3ea7d6e0f4cda0035e1d72ac0116` |
| `dbp_resnet18_1d.pt` | `0910b170881f2391ccd5fb14d742dff9ba7ea953b402f82c5fa3c852cfa05bfb` |

Use only trusted PyTorch checkpoint files. The loader preserves compatibility with the research checkpoint dictionary format. Every run records input checkpoint hashes, dataset size/mtime and paths, software versions, code hashes, and resolved configuration. Resume rejects changes to those identities. Paths are configurable, and keys are read from the environment, never written to configuration snapshots.

## Expected variation and scope

The default seed is 1337. The original calibration seed is the first eight hexadecimal digits of SHA-256 of compact JSON `[subject,target,160]`; AutoResearch, memory, and patient-only candidate seeds add 100, 200, and 300 plus candidate position. Donor seeds hash `[subject,target,recipe_id]`. Changing the master seed changes the derived seeds. Python, NumPy, and PyTorch seeds are set; deterministic cuDNN behavior is requested. Remote LLM seed support is provider-dependent.

Fresh reproduction can differ because donor adapters and development models are rebuilt, LLM responses are regenerated, proposal prompts now explicitly specify the supported 160-window budget, and timeout behavior depends on machine speed. The reported proposal generation did not set an explicit LLM seed; this repository does when supported. Development peer outcomes can also vary after refitting. Exact recovery from newly generated proposals is not promised.

The original guarded experiment was a retrospective analysis after earlier CalFree results had been inspected. Preserving the split and guard does not make that historical result an independent prospective confirmation. HITL enabled is an executable review workflow; no numerical HITL benefit is claimed.
