from __future__ import annotations
import json
import math
import numpy as np

ROUTES = ["autoresearch", "memory", "fixed_calibration", "population", "patient_only"]
TARGETS = ["SBP", "DBP"]
TIE_ATOL = 1e-9
SHAPE_KEYS = [
    "calibration_target_mean",
    "calibration_target_std",
    "ecg_std",
    "ecg_diff_std",
    "ppg_std",
    "ppg_diff_std",
]

INSTRUCTION = """You are an operational ECG/PPG personalization router. A failed
absolute error gate triggers review, NOT compulsory replacement of AutoResearch.
AutoResearch remains eligible and may be the best available model. Select the
model most likely to minimize this patient's future MAE among ALL five routes.
Retain AutoResearch if alternatives offer no supported advantage. Never select
memory automatically. Never infer future outcomes for the current patient.

You see the measured shape of fitting and validation data (distributions,
early/late and fitting/validation changes), Train-only cohort distributions,
memory donors and model provenance, and current search/routing validation MAEs.
Retrieved development examples are OTHER training patients; their internal
holdout outcomes teach route selection, not the present patient's future.
Their population model saw these patients during pretraining. Similar shape
does not guarantee transfer. Do not identify physiology from normalized amplitude
or peak-detector lag. Do not call a shift typical without a supplied reference.

The validation default combines search and routing validation evidence using a
rule selected ONLY on 48 training development patients for this target. Its
scores are ranking proxies, NOT known future errors or clinical confidence.
Review that ranking together with the actual data shape and donor evidence.
Use decision=validation_default when you cannot ground a better selection in
the supplied evidence. This is an explicit agent deferral, not a failed call.
Use decision=select to choose a route when the evidence supports your judgment.
An override of the default must identify the competing route, acknowledge both
validation errors, and cite concrete calibration or cohort evidence supporting
the override. A missed threshold alone, donor similarity alone, or a plausible
physiological story is insufficient. Read numeric rankings accurately.
Hard integrity failure requires abstain. Otherwise avoid abstaining merely for
an accuracy threshold failure: retain the best available model with limitations.
Return concise JSON, with one calibration.*, one memory.*, and candidate.<route>
evidence citation. candidate_comparison must copy the supplied exact search and
routing-validation MAEs for the proposed route and the validation default.
No test labels, test errors, prior CalFree choices or test-derived examples are
supplied. Explanations are hypotheses and are not certified diagnoses.
"""


def validate_metrics(values):
    if set(values) != set(ROUTES):
        raise ValueError("Wrong candidate bank")
    for value in values.values():
        if set(value) != {"search_MAE", "validator_MAE"} or not all(
            isinstance(x, (float, int))
            and not isinstance(x, bool)
            and math.isfinite(x)
            and x >= 0
            for x in value.values()
        ):
            raise ValueError("Invalid validation evidence")


def validation_scores(values, policy):
    validate_metrics(values)
    alpha = policy["search_weight"]
    return {
        r: alpha * values[r]["search_MAE"] + (1 - alpha) * values[r]["validator_MAE"]
        for r in ROUTES
    }


def minimum_route(scores):
    best = min(scores.values())
    # Stable numerical ties prefer retaining the already-fitted AutoResearch.
    return next(r for r in ROUTES if scores[r] <= best + TIE_ATOL)


def optimal_routes(errors):
    """Best routes (including numerical ties) on a Train peer's internal holdout."""
    best = min(errors.values())
    return [r for r in ROUTES if errors[r] <= best + TIE_ATOL]


def nearest_examples(target, shape, development):
    rows = [r for r in development["rows"] if r["target"] == target]
    x = np.array([[r["shape"][k] for k in SHAPE_KEYS] for r in rows])
    scale = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    dist = np.sqrt(
        np.mean(((x - np.array([shape[k] for k in SHAPE_KEYS])) / scale) ** 2, axis=1)
    )
    result = []
    for i in sorted(range(len(rows)), key=lambda i: (dist[i], rows[i]["subject"]))[:4]:
        r = rows[i]
        errors = r["internal_holdout_MAE"]
        result.append(
            {
                "training_subject": r["subject"],
                "target": target,
                "shape": r["shape"],
                "distance_in_development_IQR": float(dist[i]),
                "search_and_routing_validation": r["metrics"],
                "training_internal_holdout_MAE": errors,
                "best_routes_on_training_internal_holdout": optimal_routes(errors),
                "origin": "Train development patient, never a CalFree patient.",
            }
        )
    return result


def schema(facts):
    fields = {
        "decision": {
            "type": "string",
            "enum": ["select", "validation_default", "abstain"],
        },
        "route": {"type": "string", "enum": ROUTES + ["abstain"]},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(facts)},
        },
        "candidate_comparison": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                k: {"type": "number"}
                for k in [
                    "proposed_search_MAE",
                    "proposed_validator_MAE",
                    "default_search_MAE",
                    "default_validator_MAE",
                ]
            },
            "required": [
                "proposed_search_MAE",
                "proposed_validator_MAE",
                "default_search_MAE",
                "default_validator_MAE",
            ],
        },
        "data_shape_assessment": {"type": "string"},
        "rationale": {"type": "string"},
        "override_evidence": {"type": "string"},
        "limitation": {"type": "string"},
    }
    fields["calibration_evidence"] = {
        "type": "string",
        "enum": sorted(k for k in facts if k.startswith("calibration.")),
    }
    fields["memory_evidence"] = {
        "type": "string",
        "enum": sorted(k for k in facts if k.startswith("memory.")),
    }
    fields["candidate_evidence"] = {
        "type": "string",
        "enum": sorted(k for k in facts if k.startswith("candidate.")),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": fields,
        "required": list(fields),
    }


def validate_answer(answer, request):
    packet = request["packet"]
    facts = packet["facts"]
    default = packet["validation_default"]["route"]
    if not isinstance(answer, dict) or set(answer) != set(
        request["schema"]["required"]
    ):
        raise ValueError("Wrong response fields")
    decision, route = answer["decision"], answer["route"]
    if decision not in [
        "select",
        "validation_default",
        "abstain",
    ] or route not in ROUTES + ["abstain"]:
        raise ValueError("Invalid route/decision")
    if (decision == "abstain") != (route == "abstain"):
        raise ValueError("Inconsistent abstention")
    if packet["hard_integrity_failure"] and route != "abstain":
        raise ValueError("Hard integrity failure requires abstention")
    if decision == "validation_default" and route != default:
        raise ValueError("Deferral must name the actual default")
    ids = answer["evidence_ids"] + [
        answer["calibration_evidence"],
        answer["memory_evidence"],
        answer["candidate_evidence"],
    ]
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(k, str) or k not in facts for k in ids)
    ):
        raise ValueError("Invented evidence citation")
    if not all(
        any(k.startswith(prefix) for k in ids) for prefix in ["calibration.", "memory."]
    ):
        raise ValueError("Must assess calibration and cohort/memory shape")
    for key in ["data_shape_assessment", "rationale", "limitation"]:
        if not isinstance(answer[key], str) or not answer[key].strip():
            raise ValueError("Missing explanation")
    if not isinstance(answer["override_evidence"], str):
        raise ValueError("Invalid override evidence")
    if route != "abstain":
        if "candidate." + route not in ids:
            raise ValueError("Missing selected candidate citation")
        if route != default and not answer["override_evidence"].strip():
            raise ValueError("Unsupported default override")
        expected = {
            "proposed_search_MAE": facts["candidate." + route]["search_MAE"],
            "proposed_validator_MAE": facts["candidate." + route]["validator_MAE"],
            "default_search_MAE": facts["candidate." + default]["search_MAE"],
            "default_validator_MAE": facts["candidate." + default]["validator_MAE"],
        }
        actual = answer["candidate_comparison"]
        if (
            not isinstance(actual, dict)
            or set(actual) != set(expected)
            or any(
                isinstance(actual[k], bool)
                or not isinstance(actual[k], (float, int))
                or not math.isclose(actual[k], v, rel_tol=0, abs_tol=1e-6)
                for k, v in expected.items()
            )
        ):
            raise ValueError(
                "Incorrect candidate error arithmetic; copy the supplied values exactly"
            )
    return answer


def contract_reminder(packet):
    default = packet["validation_default"]["route"]
    facts = packet["facts"]
    table = {r: facts["candidate." + r] for r in ROUTES}
    return "\nCURRENT_PATIENT_DECISION_CARD:\n" + json.dumps(
        {
            "actual_default_route": default,
            "current_patient_validation": table,
            "ranking_scores": packet["validation_default"]["ranking_scores"],
            "rules": [
                "The default is exactly "
                + default
                + "; do not assume it is AutoResearch.",
                "Default comparison fields MUST use the " + default + " row above.",
                "Proposed comparison fields MUST use the row of your chosen route.",
                "Dedicated calibration_evidence, memory_evidence, and candidate_evidence fields are required.",
                "candidate_evidence must be candidate.<your selected route>.",
                "If select and default are the same route, proposed and default numbers must be identical.",
                "Select AutoResearch when its evidence is strongest; a gate failure does not forbid it.",
                "Defer to validation_default if you cannot substantiate an override.",
            ],
        },
        sort_keys=True,
    )


def gate_pass(m, t):
    a, f = m["autoresearch"], m["fixed_calibration"]
    return (
        a["validator_MAE"] <= t["mae_cap"]
        and a["validator_MAE"] <= f["validator_MAE"] * (1 + t["noninferiority"])
        and a["validator_MAE"] / max(a["search_MAE"], 1e-8) <= t["inflation_max"]
    )


def override_guard(packet, proposed, policy):
    default = packet["validation_default"]["route"]
    facts = packet["facts"]
    if packet["hard_integrity_failure"]:
        return "abstain", ["hard_integrity_failure"]
    if proposed == default:
        return default, []
    if proposed == "abstain":
        return proposed, []
    a, b = facts["candidate." + proposed], facts["candidate." + default]
    reasons = []
    if all(a[k] >= b[k] - 1e-9 for k in ["search_MAE", "validator_MAE"]) and any(
        a[k] > b[k] + 1e-9 for k in ["search_MAE", "validator_MAE"]
    ):
        reasons.append("proposed_route_dominated_on_current_validation")
    scores = packet["validation_default"]["ranking_scores"]
    if (
        scores[proposed]
        > scores[default] * (1 + policy["maximum_relative_ranking_regret"]) + 1e-9
    ):
        reasons.append("proposed_route_exceeds_validation_regret_limit")
    examples = [facts["memory.development_example_" + str(i)] for i in range(4)]
    support = sum(
        e["training_internal_holdout_MAE"][proposed]
        <= e["training_internal_holdout_MAE"][default] + 1e-9
        for e in examples
    )
    if support < policy["minimum_supporting_development_neighbors"]:
        reasons.append("insufficient_similar_training_patient_support")
    return (default if reasons else proposed), reasons
