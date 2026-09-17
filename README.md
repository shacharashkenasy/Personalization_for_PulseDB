# PulseDB reproducibility

Standalone reproduction of the 160-window PulseDB experiment. There are **three experiment modes**: `backbone`, `calibration`, and `full_pipeline`. The reported full-pipeline reference is **5.721 SBP / 3.287 DBP mean patient MAE (mmHg)** over 144 CalFree patients.



## Setup

Use Python 3.12, preferably on Linux with a CUDA GPU. CPU evaluation/training is supported but slower.

```bash
git clone https://github.com/shacharashkenasy/Personalization_for_PulseDB.git
cd Personalization_for_PulseDB
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The tested environment used PyTorch 2.10.0 with CUDA 12.8. Install a PyTorch build appropriate to your machine if necessary. Set `experiment.device: cpu` to run without CUDA.

## External data and backbone weights

```bash
export PULSEDB_DATA_PATH=/absolute/path/to/pulsedb/subsets
export PULSEDB_CHECKPOINT_PATH=/absolute/path/to/backbone/checkpoints
```

Expected layout:

```text
$PULSEDB_DATA_PATH/
  VitalDB_Train_Subset.mat          # required only for full_pipeline
  VitalDB_CalFree_Test_Subset.mat
$PULSEDB_CHECKPOINT_PATH/
  sbp_resnet18_1d.pt
  dbp_resnet18_1d.pt
```

Use the original MATLAB v7.3/HDF5 subsets, preserving patient/window order. `Subset/Signals` is `[1250, 3, N]`; channel 0 is ECG and channel 1 is PPG. **ABP is never a model input.** SBP/DBP labels and demographic fields are `[1, N]`; `Subject` and `Gender` contain MATLAB string references. Train has 465,480 windows (1,293 patients); CalFree has 57,600 windows (144 patients). No extra waveform normalization is applied.

Each checkpoint must contain `model_state_dict` for the included two-input-channel, scalar-output 1D ResNet-18. The reference backbone SHA-256 identities are listed in [reproduction details](docs/reproduction.md). **No checkpoints are bundled.** Only these two backbones are needed for a fresh run; all other models are regenerated.

You can instead set `paths.data_path` and `paths.checkpoint_path` directly in `configs/pulsedb.yaml`. Relative paths resolve from the repository directory. Each configuration needs its own `paths.output_dir`.

## Run the supported experiments

Edit the YAML and run:

```bash
python run.py --config configs/pulsedb.yaml
```

Or override individual YAML keys from the command line:

```bash
# Backbone only
python run.py --set experiment.mode=backbone \
  --set paths.output_dir=outputs/backbone

# Population calibration: the frozen population backbone, by original convention
python run.py --set experiment.mode=calibration --set calibration.type=population \
  --set paths.output_dir=outputs/calibration_population

# Fixed patient-specific fine-tuning from the backbone
python run.py --set experiment.mode=calibration --set calibration.type=finetune \
  --set paths.output_dir=outputs/calibration_finetune

# Full pipeline, automated routing
python run.py --set experiment.mode=full_pipeline --set full_pipeline.hitl=false \
  --set paths.output_dir=outputs/full_pipeline
```

`population` calibration is deliberately identical to backbone evaluation; it adds no affine fit. `finetune` trains the prediction head with the original fixed MSE recipe, using 160 fitting windows and 40 search-validation windows per patient. Neither calibration mode requires an LLM. `calibration.type` affects only calibration mode.

`--set experiment.limit=2` limits **CalFree evaluation** to the first two patients. It does not shrink the full pipeline's required Train memory/development stages. Partial-cohort metrics are marked as partial and cannot be compared to the full-cohort reference.

## Configure the LLM

Full pipeline requires a JSON-capable model endpoint. Configure:

```yaml
llm:
  provider: openai_compatible
  model: Qwen3-Coder-30B-A3B-Instruct
  base_url: http://localhost:8000/v1
  api_key_env: LLM_API_KEY
```

Use the model identifier supported by your service, and set `LLM_API_KEY` if authentication is needed. The client sends standard `/chat/completions` requests; it does not start or install a model server. For another API, set `provider: custom` and `custom_callable: package.module:function`; the function receives `(request, llm_config)` and returns a JSON object or JSON string. The package must be installed in your environment.

The agent proposes four AutoResearch recipes and four compact patient-only models, selects three retrieved Train experiences, and reviews failed-gate cases. Validated responses and prompts are logged. Proposal failures stop the run; failed routing calls fall back explicitly to the guarded validation default. `send_seed` and `json_mode` can be disabled for endpoints that do not support those parameters. Changing models/providers can change results.

## HITL and deterministic thresholds

```bash
python run.py --set full_pipeline.hitl=true \
  --set paths.output_dir=outputs/full_pipeline_hitl
```

HITL pauses **before CalFree evaluation** when failed-gate cases need review (exit code 2). Read `requests/`, `decisions/`, and the corresponding `reviews/<target>/<subject>.json`. Set `status` to `approved`, choose an available `route`, and fill in `reviewer` and `rationale`. Preserve the subject, target, and `decision_hash`. Rerun the **same command** to resume. Hard integrity failures permit only `abstain`. Gate-passing cases retain AutoResearch automatically. Human choices may change performance; the reported reference used HITL disabled.

The absolute validation-MAE thresholds are easy to change:

```yaml
full_pipeline:
  deterministic_threshold: {SBP: 8.497991049265261, DBP: 5.934352619117226}
```

A pass additionally requires the configured fixed-calibration noninferiority margin and search-to-routing error inflation limit. Changing any configuration requires a new output directory. The one supported router preserves AutoResearch as an eligible choice and guards agent overrides using validation evidence and four Train development examples.

## Results and reproducibility

| Experiment | Reference SBP MAE | Reference DBP MAE |
|---|---:|---:|
| Backbone / population calibration | 12.647 | 8.533 |
| Fine-tune calibration | 8.850 | 5.346 |
| Full pipeline, HITL disabled | **5.721** | **3.287** |

Metrics are unweighted means of patient-level MAE, RMSE, R², and bias on the final **180 windows per CalFree patient**. `summary.json` and `per_patient.csv` report metrics and coverage; per-patient JSON files also save predictions and held-out indices. Abstentions are excluded from error averages and reported through coverage.

Full pipeline first regenerates 192 Train donors' memory and 48 Train patients' development evidence, then fits models and routes CalFree patients. It saves intermediate artifacts for interruption recovery. Budget several hours or longer for a fresh full run; runtime depends on GPU, storage, and LLM service. The historical 25-second per-candidate timeout is configurable; increase it or use `null` on slower hardware. Interruptions can be resumed with identical code, inputs, configuration, and runtime.

Seeds default to 1337, with the original subject/target-derived fitting seeds. Endpoint generation, numerical kernels, candidate timeouts, and regenerated memory can still affect results. The reference is a target, not a guarantee for arbitrary models or changed thresholds. See [the original code-path trace](docs/reproduction.md) and [what was actually validated](docs/validation.md). A full fresh LLM/training reproduction is distinct from checking predictions using existing external checkpoints.

Run the self-contained tests with:

```bash
pip install pytest
python -m pytest -q
```
