"""
AI Incident Response & IT Ops Automation — OpenEnv Environment

Implements the full OpenEnv API:
  - reset(task_id) -> Observation
  - step(action)   -> (Observation, StepReward, done, info)
  - state()        -> EnvironmentState
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from graders import (
    compute_episode_score,
    grade_classification,
    grade_diagnosis,
    grade_escalation,
    grade_remediation,
    grade_resolution,
    grade_step_penalty,
    grade_wrong_action,
)
from models import (
    Action,
    ActionType,
    AlertLog,
    EpisodeResult,
    EnvironmentState,
    IncidentType,
    Observation,
    PhaseType,
    StepReward,
    SystemMetrics,
)
from tasks import TaskSpec, get_task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase → allowed actions mapping
# ---------------------------------------------------------------------------

PHASE_ACTIONS: Dict[PhaseType, List[ActionType]] = {
    PhaseType.DETECTION:      [ActionType.CLASSIFY],
    PhaseType.CLASSIFICATION: [ActionType.CLASSIFY],
    PhaseType.DIAGNOSIS:      [ActionType.DIAGNOSE],
    PhaseType.REMEDIATION:    [
        ActionType.RESTART_SERVICE,
        ActionType.CLEAR_LOGS,
        ActionType.SCALE_UP,
        ActionType.ESCALATE,
    ],
    PhaseType.VERIFICATION:   [ActionType.ESCALATE],
    PhaseType.CLOSED:         [],
}

# Diagnosis synonym tables (incident_type -> {synonym: partial_score})
DIAGNOSIS_SYNONYMS: Dict[str, Dict[str, float]] = {
    "runaway_nginx_worker": {
        "high_cpu_nginx":        0.6,
        "nginx_cpu_spike":       0.6,
        "worker_loop":           0.5,
    },
    "unrotated_log_accumulation": {
        "log_overflow":          0.6,
        "disk_log_full":         0.5,
        "logrotate_failure":     0.7,
    },
    "db_connection_pool_exhaustion": {
        "connection_pool_full":  0.7,
        "db_pool_exhausted":     0.7,
        "connection_leak":       0.5,
        "postgres_connections":  0.6,
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IncidentResponseEnv:
    """
    OpenEnv-compliant environment for IT Incident Response Automation.

    Usage:
        env = IncidentResponseEnv()
        obs = env.reset("task_easy_cpu_high")
        action = Action(action_type="classify", parameters={"incident_type": "cpu_high"})
        obs, reward, done, info = env.step(action)
    """

    def __init__(self) -> None:
        self._task: Optional[TaskSpec] = None
        self._reset_internal_state()

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _reset_internal_state(self) -> None:
        self._phase: PhaseType = PhaseType.DETECTION
        self._step_number: int = 0
        self._done: bool = False

        # Progress flags
        self._classified_correctly: bool = False
        self._classification_result: Optional[str] = None
        self._diagnosed_correctly: bool = False
        self._diagnosis_result: Optional[str] = None
        self._remediation_correct: bool = False
        self._remediation_action: Optional[str] = None
        self._escalated: bool = False
        self._resolved: bool = False

        # Reward tracking
        self._rewards_history: List[float] = []
        self._total_step_penalties: float = 0.0

        # Scores for sub-tasks
        self._classification_score: float = 0.0
        self._diagnosis_score: float = 0.0
        self._remediation_score: float = 0.0
        self._escalation_score: float = 0.0
        self._resolution_score: float = 0.0

    def _current_metrics_post_remediation(self) -> SystemMetrics:
        """Return modified metrics after successful remediation."""
        if not self._task:
            raise RuntimeError("No active task")

        base = copy.deepcopy(self._task.incident.metrics)

        if not self._remediation_correct:
            return base

        inc_type = self._task.incident.incident_type
        if inc_type == IncidentType.CPU_HIGH:
            base.cpu_usage_pct = 18.0
        elif inc_type == IncidentType.DISK_FULL:
            base.disk_usage_pct = 42.0
            base.log_size_mb = 50.0
        elif inc_type == IncidentType.SERVICE_CRASH:
            if self._task.requires_escalation and self._escalated:
                base.crashed_services = []
                base.active_services = list(
                    set(base.active_services) | {self._task.incident.affected_service}
                )
            elif not self._task.requires_escalation:
                base.crashed_services = []
                base.active_services = list(
                    set(base.active_services) | {self._task.incident.affected_service}
                )
        return base

    def _build_observation(self, status_message: str = "") -> Observation:
        if not self._task:
            raise RuntimeError("No active task")

        allowed = PHASE_ACTIONS.get(self._phase, [])
        metrics = self._current_metrics_post_remediation()

        return Observation(
            incident_id=self._task.incident.incident_id,
            phase=self._phase,
            step_number=self._step_number,
            max_steps=self._task.max_steps,
            metrics=metrics,
            recent_alerts=self._task.incident.alerts[-5:],
            incident_description=self._task.incident.description,
            classified=self._classification_result is not None,
            classification_result=self._classification_result,
            diagnosed=self._diagnosis_result is not None,
            diagnosis_result=self._diagnosis_result,
            remediation_applied=self._remediation_action is not None,
            remediation_action=self._remediation_action,
            resolved=self._resolved,
            escalated=self._escalated,
            allowed_actions=allowed,
            status_message=status_message,
        )

    def _advance_phase(self) -> None:
        """Move to next logical phase."""
        phase_order = [
            PhaseType.DETECTION,
            PhaseType.CLASSIFICATION,
            PhaseType.DIAGNOSIS,
            PhaseType.REMEDIATION,
            PhaseType.VERIFICATION,
            PhaseType.CLOSED,
        ]
        idx = phase_order.index(self._phase)
        if idx < len(phase_order) - 1:
            self._phase = phase_order[idx + 1]

    def _check_resolution(self) -> bool:
        """
        Determines if the incident is fully resolved.
        For tasks requiring escalation, both remediation AND escalation must happen.
        """
        if not self._task:
            return False
        if self._task.requires_escalation:
            return self._remediation_correct and self._escalated
        return self._remediation_correct

    # -----------------------------------------------------------------------
    # OpenEnv API
    # -----------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of the registered task IDs from tasks.py

        Returns:
            Initial Observation
        """
        self._task = get_task(task_id)
        self._reset_internal_state()
        self._phase = PhaseType.DETECTION

        logger.info(
            "Environment reset. Task=%s difficulty=%s",
            self._task.task_id,
            self._task.difficulty,
        )

        return self._build_observation(
            f"New incident detected: {self._task.incident.description[:80]}..."
        )

    def step(self, action: Action) -> Tuple[Observation, StepReward, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Args:
            action: An Action model instance

        Returns:
            (Observation, StepReward, done, info_dict)
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_number += 1

        # ---- Step penalty -----------------------------------------------
        step_pen = grade_step_penalty(self._step_number, self._task.max_steps)
        self._total_step_penalties += step_pen

        reward = StepReward(total=0.0, step_penalty=-step_pen)

        # ---- Wrong-phase penalty ----------------------------------------
        valid_for_phase = PHASE_ACTIONS.get(self._phase, [])
        wrong_pen = grade_wrong_action(
            action.action_type,
            self._phase.value,
            [a.value for a in valid_for_phase],
        )
        if wrong_pen > 0:
            reward.wrong_action_penalty = -wrong_pen
            reward.reason = (
                f"Action '{action.action_type}' not valid in phase '{self._phase.value}'. "
                f"Valid actions: {[a.value for a in valid_for_phase]}"
            )
            reward.total = round(-step_pen - wrong_pen, 4)
            self._rewards_history.append(reward.total)
            obs = self._build_observation(reward.reason)
            done = self._should_terminate()
            return obs, reward, done, self._info()

        # ---- Process action by type ------------------------------------
        action_type = ActionType(action.action_type)
        status_msg = ""

        # CLASSIFY
        if action_type == ActionType.CLASSIFY:
            predicted = action.parameters.get("incident_type", "")
            score = grade_classification(predicted, self._task.correct_classification)
            self._classification_score = score
            self._classification_result = predicted
            if score == 1.0:
                self._classified_correctly = True
                reward.classification_reward = 0.20
                status_msg = f"✓ Correct classification: {predicted}"
            else:
                reward.classification_reward = 0.0
                status_msg = f"✗ Incorrect classification: '{predicted}'"
            self._advance_phase()  # always advance past classification

        # DIAGNOSE
        elif action_type == ActionType.DIAGNOSE:
            predicted = action.parameters.get("root_cause", "")
            synonyms = DIAGNOSIS_SYNONYMS.get(self._task.correct_diagnosis, {})
            score = grade_diagnosis(predicted, self._task.correct_diagnosis, synonyms)
            self._diagnosis_score = score
            self._diagnosis_result = predicted
            if score >= 1.0:
                self._diagnosed_correctly = True
                reward.diagnosis_reward = 0.25
                status_msg = f"✓ Correct diagnosis: {predicted}"
            elif score > 0:
                self._diagnosed_correctly = False
                reward.diagnosis_reward = round(0.25 * score, 4)
                status_msg = f"~ Partial diagnosis credit ({score:.2f}): {predicted}"
            else:
                reward.diagnosis_reward = 0.0
                status_msg = f"✗ Incorrect diagnosis: '{predicted}'"
            self._advance_phase()

        # RESTART_SERVICE
        elif action_type == ActionType.RESTART_SERVICE:
            score = grade_remediation(
                ActionType.RESTART_SERVICE.value,
                self._task.correct_remediation,
                phase_correct=True,
            )
            self._remediation_score = score
            self._remediation_action = ActionType.RESTART_SERVICE.value
            if score == 1.0:
                self._remediation_correct = True
                reward.remediation_reward = 0.30
                status_msg = "✓ Service restarted successfully"
            else:
                reward.remediation_reward = -0.05
                status_msg = "✗ Restart was not the correct remediation for this incident"
            self._advance_phase()

        # CLEAR_LOGS
        elif action_type == ActionType.CLEAR_LOGS:
            score = grade_remediation(
                ActionType.CLEAR_LOGS.value,
                self._task.correct_remediation,
                phase_correct=True,
            )
            self._remediation_score = score
            self._remediation_action = ActionType.CLEAR_LOGS.value
            if score == 1.0:
                self._remediation_correct = True
                reward.remediation_reward = 0.30
                status_msg = "✓ Logs cleared — disk space recovered"
            else:
                reward.remediation_reward = -0.05
                status_msg = "✗ Clearing logs was not the correct remediation"
            self._advance_phase()

        # SCALE_UP
        elif action_type == ActionType.SCALE_UP:
            score = grade_remediation(
                ActionType.SCALE_UP.value,
                self._task.correct_remediation,
                phase_correct=True,
            )
            self._remediation_score = score
            self._remediation_action = ActionType.SCALE_UP.value
            if score == 1.0:
                self._remediation_correct = True
                reward.remediation_reward = 0.30
                status_msg = "✓ Service scaled up — load distributed"
            else:
                reward.remediation_reward = -0.05
                status_msg = "✗ Scaling up was not the correct remediation"
            self._advance_phase()

        # ESCALATE
        elif action_type == ActionType.ESCALATE:
            self._escalated = True
            esc_score = grade_escalation(
                escalated=True,
                required=self._task.requires_escalation,
            )
            self._escalation_score = esc_score
            if self._task.requires_escalation:
                reward.escalation_reward = 0.15
                status_msg = "✓ Correctly escalated to senior team"
            else:
                reward.escalation_reward = -0.05
                status_msg = "~ Unnecessary escalation (minor penalty)"
            # Escalate can happen in REMEDIATION or VERIFICATION
            if self._phase == PhaseType.REMEDIATION and not self._remediation_correct:
                pass  # stay in remediation — must still apply fix
            else:
                self._advance_phase()

        # ---- Check for resolution ---------------------------------------
        self._resolved = self._check_resolution()
        if self._resolved:
            res_score = grade_resolution(
                resolved=True,
                classified_correctly=self._classified_correctly,
                diagnosed_correctly=self._diagnosed_correctly,
                remediation_correct=self._remediation_correct,
            )
            self._resolution_score = res_score
            reward.resolution_reward = round(0.25 * res_score, 4)
            self._phase = PhaseType.CLOSED
            status_msg += " | Incident RESOLVED ✓"

        # ---- Total reward -----------------------------------------------
        gross = (
            reward.classification_reward
            + reward.diagnosis_reward
            + reward.remediation_reward
            + reward.resolution_reward
            + reward.escalation_reward
        )
        reward.total = round(gross - step_pen + reward.wrong_action_penalty, 4)
        reward.reason = status_msg
        self._rewards_history.append(reward.total)

        done = self._should_terminate()
        if done:
            self._done = True
            self._phase = PhaseType.CLOSED

        obs = self._build_observation(status_msg)
        return obs, reward, done, self._info()

    def state(self) -> EnvironmentState:
        """Return a full snapshot of the internal environment state."""
        if self._task is None:
            raise RuntimeError("Call reset() before state().")
        return EnvironmentState(
            task_id=self._task.task_id,
            incident=self._task.incident,
            current_phase=self._phase,
            step_number=self._step_number,
            classified_correctly=self._classified_correctly,
            diagnosed_correctly=self._diagnosed_correctly,
            remediation_correct=self._remediation_correct,
            resolved=self._resolved,
            escalated=self._escalated,
            rewards_history=list(self._rewards_history),
            done=self._done,
        )

    def compute_final_score(self) -> EpisodeResult:
        """Compute and return the normalised episode score."""
        if self._task is None:
            raise RuntimeError("Call reset() before compute_final_score().")

        score = compute_episode_score(
            weights=self._task.reward_weights,
            classification_score=self._classification_score,
            diagnosis_score=self._diagnosis_score,
            remediation_score=self._remediation_score,
            resolution_score=self._resolution_score,
            escalation_score=self._escalation_score if self._task.requires_escalation else 1.0,
            total_step_penalties=self._total_step_penalties,
        )

        return EpisodeResult(
            incident_id=self._task.incident.incident_id,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            success=self._resolved,
            score=score,
            total_steps=self._step_number,
            rewards_per_step=list(self._rewards_history),
            cumulative_reward=round(sum(self._rewards_history), 4),
            phase_reached=self._phase,
            details={
                "classified_correctly": self._classified_correctly,
                "diagnosed_correctly":  self._diagnosed_correctly,
                "remediation_correct":  self._remediation_correct,
                "escalated":            self._escalated,
                "resolved":             self._resolved,
                "classification_score": self._classification_score,
                "diagnosis_score":      self._diagnosis_score,
                "remediation_score":    self._remediation_score,
                "resolution_score":     self._resolution_score,
                "escalation_score":     self._escalation_score,
                "total_step_penalties": self._total_step_penalties,
            },
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _should_terminate(self) -> bool:
        if self._phase == PhaseType.CLOSED:
            return True
        if self._step_number >= self._task.max_steps:  # type: ignore[union-attr]
            return True
        return False

    def _info(self) -> Dict[str, Any]:
        return {
            "task_id":   self._task.task_id if self._task else None,  # type: ignore[union-attr]
            "phase":     self._phase.value,
            "step":      self._step_number,
            "resolved":  self._resolved,
            "escalated": self._escalated,
        }
