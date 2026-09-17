"""Scientific split, routing, review, and provider-boundary regression tests."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import pytest
from pulsedb.config import ROOT, read, load_config, atomic_json, stable_hash
from pulsedb.data import PulseDBFile, RestrictedData, personalization_splits
from pulsedb.experiment import review_decisions, ReviewRequired, aggregate
from pulsedb.llm import Agent
from pulsedb.routing import (
    ROUTES,
    gate_pass,
    override_guard,
    minimum_route,
    schema,
    validate_answer,
)


def test_real_matlab_channel_layout_and_partition(tmp_path):
    import h5py

    path = tmp_path / "tiny.mat"
    with h5py.File(path, "w") as f:
        g = f.create_group("Subset")
        text = f.create_dataset(
            "patient_name", data=np.array([ord(c) for c in "patient"])
        )
        refs = g.create_dataset("Subject", (1, 400), dtype=h5py.ref_dtype)
        refs[0, :] = [text.ref] * 400
        x = g.create_dataset("Signals", (8, 3, 400), dtype="f4")
        x[:, 0, :], x[:, 1, :], x[:, 2, :] = 1, 2, 999
        g.create_dataset("SBP", data=np.full((1, 400), 120.0))
    with PulseDBFile(path, "SBP") as mat:
        np.testing.assert_array_equal(mat.signal(0), [[1] * 8, [2] * 8])
        (split,) = personalization_splits(mat, "calfree")
        assert list(
            map(
                len,
                [
                    split.acquisition_pool,
                    split.search_validation,
                    split.validator,
                    split.final_holdout,
                ],
            )
        ) == [160, 40, 20, 180]
        assert set(split.final_holdout) == set(range(220, 400))
        restricted = RestrictedData(
            mat, split.acquisition_pool + split.search_validation + split.validator
        )
        for getter in (restricted.signal, restricted.label):
            with pytest.raises(PermissionError):
                getter(220)
        assert restricted.label(219) == 120


def packet():
    facts = {
        "candidate." + r: {"search_MAE": 5.0, "validator_MAE": 5.0} for r in ROUTES
    }
    facts["candidate.memory"] = {"search_MAE": 4.9, "validator_MAE": 5.1}
    for i in range(4):
        facts["memory.development_example_" + str(i)] = {
            "training_internal_holdout_MAE": {
                r: 4.0 if r == "memory" else 5.0 for r in ROUTES
            }
        }
    facts["calibration.fitting"] = {"label": {"mean": 120.0}}
    return {
        "hard_integrity_failure": False,
        "facts": facts,
        "validation_default": {
            "route": "autoresearch",
            "ranking_scores": {r: 5.0 for r in ROUTES},
        },
    }


POLICY = {
    "maximum_relative_ranking_regret": 0.05,
    "minimum_supporting_development_neighbors": 3,
}


def test_gate_requires_all_three_conditions_and_inclusive_boundary():
    t = {"mae_cap": 8, "noninferiority": 0.1, "inflation_max": 1.5}
    m = {
        "autoresearch": {"search_MAE": 6, "validator_MAE": 8},
        "fixed_calibration": {"validator_MAE": 8},
    }
    assert gate_pass(m, t)
    for change in (
        {"mae_cap": 7.99},
        {"inflation_max": 1.3},
        {"noninferiority": -0.01},
    ):
        assert not gate_pass(m, t | change)


def test_guard_retention_override_dominance_and_training_support():
    p = packet()
    assert override_guard(p, "autoresearch", POLICY) == ("autoresearch", [])
    assert override_guard(p, "memory", POLICY) == ("memory", [])
    p["facts"]["candidate.memory"] = {"search_MAE": 5.1, "validator_MAE": 5.1}
    assert override_guard(p, "memory", POLICY)[0] == "autoresearch"
    p = packet()
    for i in range(2):
        p["facts"]["memory.development_example_" + str(i)][
            "training_internal_holdout_MAE"
        ]["memory"] = 6
    assert override_guard(p, "memory", POLICY)[0] == "autoresearch"
    p["hard_integrity_failure"] = True
    assert override_guard(p, "autoresearch", POLICY)[0] == "abstain"
    assert minimum_route({r: 5.0 for r in ROUTES}) == "autoresearch"


def test_router_rejects_wrong_numeric_evidence():
    p = packet()
    request = {"packet": p, "schema": schema(p["facts"])}
    answer = {
        "decision": "validation_default",
        "route": "autoresearch",
        "evidence_ids": [
            "calibration.fitting",
            "memory.development_example_0",
            "candidate.autoresearch",
        ],
        "calibration_evidence": "calibration.fitting",
        "memory_evidence": "memory.development_example_0",
        "candidate_evidence": "candidate.autoresearch",
        "data_shape_assessment": "Observed fitting distribution.",
        "rationale": "No supported override.",
        "override_evidence": "",
        "limitation": "Future error unknown.",
        "candidate_comparison": dict.fromkeys(
            [
                "proposed_search_MAE",
                "proposed_validator_MAE",
                "default_search_MAE",
                "default_validator_MAE",
            ],
            5.0,
        ),
    }
    assert validate_answer(answer, request) == answer
    answer["candidate_comparison"]["default_search_MAE"] = 4.0
    with pytest.raises(ValueError, match="arithmetic"):
        validate_answer(answer, request)


def test_hitl_pending_resume_stale_and_hard_failure(tmp_path):
    cfg = {"full_pipeline": {"hitl": True}}
    choice = {
        "subject": "p1",
        "target": "SBP",
        "route": "memory",
        "gate_passed": False,
        "hard_integrity_failure": False,
    }
    with pytest.raises(ReviewRequired):
        review_decisions(cfg, tmp_path, [choice])
    assert not (tmp_path / "results").exists()
    path = tmp_path / "reviews/SBP/p1.json"
    r = read(path)
    r.update(
        status="approved",
        reviewer="Researcher",
        rationale="Reviewed validation evidence.",
        route="autoresearch",
    )
    atomic_json(path, r)
    assert review_decisions(cfg, tmp_path, [choice])[0]["route"] == "autoresearch"
    with pytest.raises(ValueError, match="Stale"):
        review_decisions(cfg, tmp_path, [choice | {"route": "population"}])
    hard = choice | {"hard_integrity_failure": True, "route": "abstain"}
    r.update(decision_hash=stable_hash(hard), route="memory")
    atomic_json(path, r)
    with pytest.raises(ValueError, match="unsafe"):
        review_decisions(cfg, tmp_path, [hard])
    assert review_decisions({"full_pipeline": {"hitl": False}}, tmp_path, [choice]) == [
        choice
    ]


def test_generic_http_provider_retry_and_audit(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            content = "not json" if len(requests) == 1 else '{"ok": true}'
            body = json.dumps(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": content}}
                    ]
                }
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg = {
        "provider": "openai_compatible",
        "model": "test-model",
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "api_key_env": "UNUSED_TEST_API_KEY",
        "send_seed": True,
        "json_mode": True,
        "proposal_attempts": 2,
        "routing_attempts": 3,
        "proposal_temperature": 0.2,
        "routing_temperature": 0,
        "max_tokens": 64,
        "timeout_seconds": 5,
    }
    try:
        agent = Agent(cfg, tmp_path / "audit.jsonl")
        assert agent.call(
            kind="test", prompt="Return JSON", evidence={}, validator=lambda x: x
        ) == {"ok": True}
        assert len(requests) == 2 and requests[-1]["seed"] == 1337
        logs = [
            json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()
        ]
        assert [v["status"] for v in logs] == ["failed", "complete"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_aggregation_is_patient_weighted_and_reports_abstention(tmp_path):
    cfg = {
        "experiment": {"mode": "full_pipeline", "targets": ["SBP"], "limit": None},
        "calibration": {"type": "population"},
    }
    rows = [
        {
            "subject": "a",
            "target": "SBP",
            "route": "autoresearch",
            "covered": True,
            "metrics": {"MAE": 2.0, "RMSE": 3.0, "R2": 0.1, "bias": 1.0, "n": 1},
        },
        {
            "subject": "b",
            "target": "SBP",
            "route": "memory",
            "covered": True,
            "metrics": {"MAE": 8.0, "RMSE": 9.0, "R2": 0.3, "bias": -1.0, "n": 100},
        },
        {
            "subject": "c",
            "target": "SBP",
            "route": "abstain",
            "covered": False,
            "metrics": None,
        },
    ]
    summary = aggregate(tmp_path, rows, cfg)["targets"]["SBP"]
    assert summary["MAE"] == 5.0 and summary["coverage"] == 2 / 3


def test_protocol_is_train_only_and_config_exposes_only_three_modes(tmp_path):
    protocol = read(ROOT / "configs/protocol.json")
    assert len(protocol["calfree_subjects"]) == 144
    assert len(protocol["memory_subjects"]) == 192
    assert len(protocol["development_subjects"]) == 48
    assert set(protocol["development_subjects"]) <= set(protocol["memory_subjects"])
    assert not set(protocol["memory_subjects"]) & set(protocol["calfree_subjects"])
    overrides = [f"paths.data_path={tmp_path}", f"paths.checkpoint_path={tmp_path}"]
    for mode in ["backbone", "calibration", "full_pipeline"]:
        assert (
            load_config(
                ROOT / "configs/pulsedb.yaml", overrides + ["experiment.mode=" + mode]
            )["experiment"]["mode"]
            == mode
        )
    with pytest.raises(ValueError):
        load_config(
            ROOT / "configs/pulsedb.yaml", overrides + ["experiment.mode=random_hpo"]
        )


def test_full_driver_waits_for_review_before_any_test_access(tmp_path, monkeypatch):
    """Exercise the actual full-mode controller through pause and resume.

    Expensive model fitting is replaced here; review, routing guard, freezing,
    output writing, and evaluation ordering use the production controller.
    """
    from pulsedb import experiment as e
    from pulsedb.data import PersonalizationSplit

    cfg = load_config(
        ROOT / "configs/pulsedb.yaml",
        [
            f"paths.data_path={tmp_path}",
            f"paths.checkpoint_path={tmp_path}",
            f"paths.output_dir={tmp_path}/run",
            "experiment.device=cpu",
            "experiment.targets=[SBP]",
            "full_pipeline.hitl=true",
        ],
    )
    out = tmp_path / "run"
    out.mkdir()
    split = PersonalizationSplit("fixture", (0,), (1,), (2,), (3, 4), "locked_test")

    class Mat:
        target = "SBP"

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class TestAgent:
        def __init__(self, *args):
            pass

        def check_ready(self):
            pass

        def call(self, **kwargs):
            p = kwargs["evidence"]
            r = p["validation_default"]["route"]
            m = p["facts"]["candidate." + r]
            return kwargs["validator"](
                {
                    "decision": "validation_default",
                    "route": r,
                    "evidence_ids": [
                        "calibration.fitting",
                        "memory.development_example_0",
                        "candidate." + r,
                    ],
                    "calibration_evidence": "calibration.fitting",
                    "memory_evidence": "memory.development_example_0",
                    "candidate_evidence": "candidate." + r,
                    "data_shape_assessment": "Fixture observations.",
                    "rationale": "Fixture deferral.",
                    "override_evidence": "",
                    "limitation": "Test fixture.",
                    "candidate_comparison": {
                        "proposed_search_MAE": m["search_MAE"],
                        "proposed_validator_MAE": m["validator_MAE"],
                        "default_search_MAE": m["search_MAE"],
                        "default_validator_MAE": m["validator_MAE"],
                    },
                }
            )

    p = packet()
    p["facts"]["candidate.autoresearch"]["validator_MAE"] = 9.0
    bundle = {
        "artifacts": {},
        "metrics": {r: p["facts"]["candidate." + r] for r in ROUTES},
        "integrity": {"hard_integrity_failure": False},
    }

    def fitted(*args):
        atomic_json(out / "candidates/calfree/SBP/fixture/bundle.json", bundle)
        return bundle

    evaluations = []

    def evaluated(*args):
        assert (out / "frozen_decisions.json").exists()
        evaluations.append(True)
        return {
            "covered": True,
            "predictions": [1, 2],
            "indices": [3, 4],
            "metrics": {"MAE": 1.0, "RMSE": 1.0, "R2": 0.0, "bias": 0.0, "n": 2},
        }

    monkeypatch.setattr(e, "initialize", lambda *args: None)
    monkeypatch.setattr(e, "PulseDBFile", Mat)
    monkeypatch.setattr(e, "subject_splits", lambda *args: [split])
    monkeypatch.setattr(e, "Agent", TestAgent)
    monkeypatch.setattr(e, "build_memory", lambda *args: [])
    monkeypatch.setattr(e, "prepare_development", lambda *args: {})
    monkeypatch.setattr(e, "fit_bank", fitted)
    monkeypatch.setattr(e, "make_packet", lambda *args: p)
    monkeypatch.setattr(e, "evaluate_route", evaluated)
    with pytest.raises(ReviewRequired):
        e.run(cfg)
    assert not evaluations and not (out / "frozen_decisions.json").exists()
    review_path = out / "reviews/SBP/fixture.json"
    review = read(review_path) | {
        "status": "approved",
        "reviewer": "Reviewer",
        "rationale": "Checked fixture.",
    }
    atomic_json(review_path, review)
    summary = e.run(cfg)
    assert evaluations == [True] and summary["targets"]["SBP"]["coverage"] == 1
    # A later route edit cannot alter choices after the evaluation boundary.
    atomic_json(review_path, review | {"route": "memory"})
    with pytest.raises(RuntimeError, match="after held-out"):
        e.run(cfg)
    assert evaluations == [True]


def test_fitting_cache_preserves_values_and_does_not_expose_test_windows():
    from types import SimpleNamespace

    reads = []
    source = np.arange(12, dtype=np.float32).reshape(6, 2).T

    def signal(idx):
        reads.append(idx)
        return source.copy(order="K")

    mat = SimpleNamespace(target="SBP", signal=signal, label=lambda i, t: 123.0)
    view = RestrictedData(mat, [0, 1])
    np.testing.assert_array_equal(view.signal(0), source)
    modified = view.signal(0)
    modified[0, 0] = 999
    np.testing.assert_array_equal(view.signal(0), source)
    assert reads == [0]
    with pytest.raises(PermissionError):
        view.signal(2)
    assert reads == [0]
