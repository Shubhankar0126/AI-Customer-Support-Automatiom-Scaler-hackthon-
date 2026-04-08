"""
Typed Pydantic models for AI Incident Response & IT Ops Automation OpenEnv.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class IncidentType(str, Enum):
    CPU_HIGH = "cpu_high"
    DISK_FULL = "disk_full"
    SERVICE_CRASH = "service_crash"


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    DIAGNOSE = "diagnose"
    RESTART_SERVICE = "restart_service"
    CLEAR_LOGS = "clear_logs"
    SCALE_UP = "scale_up"
    ESCALATE = "escalate"


class PhaseType(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    DIAGNOSIS = "diagnosis"
    REMEDIATION = "remediation"
    VERIFICATION = "verification"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class SystemMetrics(BaseModel):
    cpu_usage_pct: float = Field(..., ge=0.0, le=100.0, description="CPU utilisation %")
    disk_usage_pct: float = Field(..., ge=0.0, le=100.0, description="Disk utilisation %")
    memory_usage_pct: float = Field(..., ge=0.0, le=100.0, description="Memory utilisation %")
    active_services: List[str] = Field(default_factory=list)
    crashed_services: List[str] = Field(default_factory=list)
    log_size_mb: float = Field(default=0.0, ge=0.0)


class AlertLog(BaseModel):
    timestamp: str
    level: str  # INFO | WARN | ERROR | CRITICAL
    source: str
    message: str


class IncidentContext(BaseModel):
    incident_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    affected_service: str
    metrics: SystemMetrics
    alerts: List[AlertLog]
    description: str


# ---------------------------------------------------------------------------
# OpenEnv API models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent observes at each step."""
    incident_id: str
    phase: PhaseType
    step_number: int
    max_steps: int

    # Current system state
    metrics: SystemMetrics
    recent_alerts: List[AlertLog]
    incident_description: str

    # Progress tracking (visible to agent)
    classified: bool = False
    classification_result: Optional[str] = None
    diagnosed: bool = False
    diagnosis_result: Optional[str] = None
    remediation_applied: bool = False
    remediation_action: Optional[str] = None
    resolved: bool = False

    # Escalation flag
    escalated: bool = False

    # Allowed actions in current phase
    allowed_actions: List[ActionType]

    # Human-readable status
    status_message: str = ""

    class Config:
        use_enum_values = True


class Action(BaseModel):
    """Action submitted by the agent."""
    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class StepReward(BaseModel):
    """Granular reward breakdown for a single step."""
    total: float = Field(..., ge=-1.0, le=1.0)

    # Positive components
    classification_reward: float = 0.0
    diagnosis_reward: float = 0.0
    remediation_reward: float = 0.0
    resolution_reward: float = 0.0
    escalation_reward: float = 0.0

    # Negative components
    wrong_action_penalty: float = 0.0
    step_penalty: float = 0.0

    reason: str = ""


class EpisodeResult(BaseModel):
    """Final result after the episode ends."""
    incident_id: str
    task_id: str
    difficulty: str
    success: bool
    score: float = Field(..., ge=0.0, le=1.0)
    total_steps: int
    rewards_per_step: List[float]
    cumulative_reward: float
    phase_reached: PhaseType
    details: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class EnvironmentState(BaseModel):
    """Full internal state snapshot (for debugging / logging)."""
    task_id: str
    incident: IncidentContext
    current_phase: PhaseType
    step_number: int
    classified_correctly: bool
    diagnosed_correctly: bool
    remediation_correct: bool
    resolved: bool
    escalated: bool
    rewards_history: List[float]
    done: bool

    class Config:
        use_enum_values = True
