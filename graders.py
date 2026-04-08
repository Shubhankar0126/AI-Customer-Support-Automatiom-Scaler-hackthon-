"""
Deterministic grading functions for AI Incident Response & IT Ops Automation.

Each grader returns a float in [0.0, 1.0] for its sub-task.
All comparisons are case-insensitive and normalised.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from models import ActionType, IncidentType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Individual graders
# ---------------------------------------------------------------------------

def grade_classification(
    predicted: Optional[str],
    expected: str,
) -> float:
    """
    Returns 1.0 if the agent correctly classifies the incident type, else 0.0.
    Partial credit is not given for classification — it is binary.
    """
    if predicted is None:
        return 0.0
    if _normalise(predicted) == _normalise(expected):
        return 1.0
    return 0.0


def grade_diagnosis(
    predicted: Optional[str],
    expected: str,
    synonyms: Optional[Dict[str, float]] = None,
) -> float:
    """
    Returns 1.0 for exact match, partial credit for near-matches via synonyms.

    synonyms: mapping of alternative string → partial score (0 < score < 1)
    Example: {"db_pool_exhausted": 0.7, "connection_leak": 0.5}
    """
    if predicted is None:
        return 0.0
    pred_norm = _normalise(predicted)
    exp_norm = _normalise(expected)

    if pred_norm == exp_norm:
        return 1.0

    # Check synonym table for partial credit
    if synonyms:
        for synonym, partial in synonyms.items():
            if pred_norm == _normalise(synonym):
                return float(partial)

    return 0.0


def grade_remediation(
    predicted_action: Optional[str],
    expected_action: ActionType,
    phase_correct: bool = True,
) -> float:
    """
    Returns 1.0 if the agent applied the correct remediation action.
    Returns 0.0 if the action is wrong or phase is incorrect.
    """
    if predicted_action is None or not phase_correct:
        return 0.0
    if _normalise(predicted_action) == _normalise(expected_action.value):
        return 1.0
    return 0.0


def grade_escalation(
    escalated: bool,
    required: bool,
) -> float:
    """
    Returns 1.0 if escalation matches the requirement.
    Returns 0.5 if escalation was unnecessary but performed (minor penalty).
    Returns 0.0 if escalation was required but not performed.
    """
    if required and escalated:
        return 1.0
    if not required and not escalated:
        return 1.0       # correctly chose NOT to escalate
    if not required and escalated:
        return 0.5       # unnecessary escalation — minor penalty
    # required and not escalated
    return 0.0


def grade_resolution(
    resolved: bool,
    classified_correctly: bool,
    diagnosed_correctly: bool,
    remediation_correct: bool,
) -> float:
    """
    Resolution score reflects end-to-end closure.
    Full score only when all preceding stages are correct AND resolved.
    Partial credit for partial pipeline completion even without full resolution.
    """
    if resolved and classified_correctly and diagnosed_correctly and remediation_correct:
        return 1.0

    # Partial credit: each correct stage contributes
    score = 0.0
    if classified_correctly:
        score += 0.25
    if diagnosed_correctly:
        score += 0.35
    if remediation_correct:
        score += 0.40
    return min(score, 0.95)   # cap partial at 0.95 — never full without resolution


def grade_step_penalty(
    step_number: int,
    max_steps: int,
    base_penalty: float = 0.02,
) -> float:
    """
    Returns a small negative penalty per step to encourage efficiency.
    Penalty is capped so it cannot dominate the reward signal.
    """
    # Linear scaling: first step has smallest penalty
    fraction = step_number / max(max_steps, 1)
    penalty = base_penalty * (1.0 + fraction)
    return round(min(penalty, 0.05), 4)   # cap at -0.05/step


def grade_wrong_action(
    action_type: str,
    current_phase: str,
    valid_actions_for_phase: list,
) -> float:
    """
    Returns a penalty (positive float, applied as negative) when the agent
    submits an action that is invalid for the current phase.
    """
    if _normalise(action_type) in [_normalise(a) for a in valid_actions_for_phase]:
        return 0.0   # no penalty
    return 0.10      # wrong-phase action penalty


# ---------------------------------------------------------------------------
# Composite episode scorer
# ---------------------------------------------------------------------------

def compute_episode_score(
    weights: Dict[str, float],
    classification_score: float,
    diagnosis_score: float,
    remediation_score: float,
    resolution_score: float,
    escalation_score: float = 1.0,
    total_step_penalties: float = 0.0,
) -> float:
    """
    Computes a normalised episode score in [0.0, 1.0].

    weights: task-specific weights dict (keys: classification, diagnosis,
             remediation, resolution, escalation)
    """
    # Build weighted sum
    raw = (
        weights.get("classification", 0.20) * classification_score
        + weights.get("diagnosis",      0.25) * diagnosis_score
        + weights.get("remediation",    0.30) * remediation_score
        + weights.get("resolution",     0.25) * resolution_score
        + weights.get("escalation",     0.00) * escalation_score
    )

    # Re-normalise weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        raw = raw / total_weight

    # Apply step penalties
    penalised = raw - total_step_penalties
    return float(max(0.0, min(1.0, penalised)))
