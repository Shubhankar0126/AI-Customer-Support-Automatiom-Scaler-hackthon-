"""
Task definitions for AI Incident Response & IT Ops Automation OpenEnv.
Each task specifies a concrete incident scenario with deterministic parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import (
    ActionType,
    AlertLog,
    IncidentContext,
    IncidentSeverity,
    IncidentType,
    PhaseType,
    SystemMetrics,
)


# ---------------------------------------------------------------------------
# Task specification dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    task_id: str
    difficulty: str                    # easy | medium | hard
    name: str
    description: str
    incident: IncidentContext
    correct_classification: str        # expected IncidentType value
    correct_diagnosis: str             # expected root-cause string
    correct_remediation: ActionType    # the action that fixes the incident
    requires_escalation: bool = False  # hard tasks may require escalation
    max_steps: int = 10
    # Per-task reward weights (sum to 1.0 before step penalty)
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "classification": 0.20,
        "diagnosis": 0.25,
        "remediation": 0.30,
        "resolution": 0.25,
    })


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

def _make_alerts(items: List[tuple]) -> List[AlertLog]:
    return [AlertLog(timestamp=ts, level=lvl, source=src, message=msg)
            for ts, lvl, src, msg in items]


# -- EASY: cpu_high ----------------------------------------------------------

TASK_EASY = TaskSpec(
    task_id="task_easy_cpu_high",
    difficulty="easy",
    name="High CPU Usage — Web Server",
    description=(
        "The monitoring system reports that the production web server 'nginx' "
        "is consuming 94% CPU. Response times have degraded. "
        "Classify the incident, diagnose the root cause, and apply the correct fix."
    ),
    incident=IncidentContext(
        incident_id="INC-001",
        incident_type=IncidentType.CPU_HIGH,
        severity=IncidentSeverity.HIGH,
        affected_service="nginx",
        metrics=SystemMetrics(
            cpu_usage_pct=94.2,
            disk_usage_pct=45.0,
            memory_usage_pct=62.0,
            active_services=["nginx", "postgres", "redis"],
            crashed_services=[],
            log_size_mb=120.0,
        ),
        alerts=_make_alerts([
            ("2024-01-15T10:00:00Z", "CRITICAL", "prometheus", "CPU usage exceeded 90% threshold on web-01"),
            ("2024-01-15T10:01:00Z", "ERROR",    "nginx",      "Worker process CPU time 98% – possible runaway loop"),
            ("2024-01-15T10:02:00Z", "WARN",     "loadbalancer","Upstream response time > 5s"),
        ]),
        description=(
            "Web server CPU has been at 94%+ for 10 minutes. "
            "Multiple nginx worker processes are consuming CPU. "
            "Traffic volume is normal. No memory pressure observed."
        ),
    ),
    correct_classification=IncidentType.CPU_HIGH.value,
    correct_diagnosis="runaway_nginx_worker",
    correct_remediation=ActionType.RESTART_SERVICE,
    requires_escalation=False,
    max_steps=8,
    reward_weights={
        "classification": 0.20,
        "diagnosis":      0.25,
        "remediation":    0.30,
        "resolution":     0.25,
    },
)


# -- MEDIUM: disk_full -------------------------------------------------------

TASK_MEDIUM = TaskSpec(
    task_id="task_medium_disk_full",
    difficulty="medium",
    name="Disk Full — Application Logs",
    description=(
        "The application server 'app-01' has reached 98% disk utilisation. "
        "The primary culprit is unrotated application log files in /var/log/app. "
        "Classify, diagnose, apply the correct remediation, and verify resolution."
    ),
    incident=IncidentContext(
        incident_id="INC-002",
        incident_type=IncidentType.DISK_FULL,
        severity=IncidentSeverity.CRITICAL,
        affected_service="app-service",
        metrics=SystemMetrics(
            cpu_usage_pct=35.0,
            disk_usage_pct=98.1,
            memory_usage_pct=55.0,
            active_services=["app-service", "postgres"],
            crashed_services=[],
            log_size_mb=47500.0,   # ~47 GB of logs
        ),
        alerts=_make_alerts([
            ("2024-01-15T11:00:00Z", "CRITICAL", "disk-monitor",   "Disk usage at 98% on /dev/sda1 (app-01)"),
            ("2024-01-15T11:01:00Z", "ERROR",    "app-service",    "Cannot write to /var/log/app/app.log: No space left on device"),
            ("2024-01-15T11:02:00Z", "ERROR",    "app-service",    "Transaction log write failure – DB state may be inconsistent"),
            ("2024-01-15T11:03:00Z", "WARN",     "logrotate",      "logrotate did not run for 7 days — cron misconfiguration suspected"),
        ]),
        description=(
            "Disk at 98% on app-01. Log files in /var/log/app total ~47 GB. "
            "logrotate has not run for 7 days due to a cron misconfiguration. "
            "Application writes are failing. Database consistency risk if not resolved quickly."
        ),
    ),
    correct_classification=IncidentType.DISK_FULL.value,
    correct_diagnosis="unrotated_log_accumulation",
    correct_remediation=ActionType.CLEAR_LOGS,
    requires_escalation=False,
    max_steps=10,
    reward_weights={
        "classification": 0.20,
        "diagnosis":      0.25,
        "remediation":    0.30,
        "resolution":     0.25,
    },
)


# -- HARD: service_crash with escalation required ----------------------------

TASK_HARD = TaskSpec(
    task_id="task_hard_service_crash",
    difficulty="hard",
    name="Cascading Service Crash — Payment Gateway",
    description=(
        "The payment-gateway service has crashed repeatedly (3 times in 30 min). "
        "Restart attempts fail due to an underlying database connection pool exhaustion. "
        "The issue requires service restart AND escalation to the database team. "
        "Correct sequence: classify → diagnose → restart_service → escalate → verify."
    ),
    incident=IncidentContext(
        incident_id="INC-003",
        incident_type=IncidentType.SERVICE_CRASH,
        severity=IncidentSeverity.CRITICAL,
        affected_service="payment-gateway",
        metrics=SystemMetrics(
            cpu_usage_pct=22.0,
            disk_usage_pct=60.0,
            memory_usage_pct=88.0,
            active_services=["api-gateway", "user-service", "postgres"],
            crashed_services=["payment-gateway"],
            log_size_mb=300.0,
        ),
        alerts=_make_alerts([
            ("2024-01-15T12:00:00Z", "CRITICAL", "k8s",             "payment-gateway CrashLoopBackOff — restart count: 3"),
            ("2024-01-15T12:01:00Z", "ERROR",    "payment-gateway", "Failed to acquire DB connection: pool exhausted (max=50)"),
            ("2024-01-15T12:02:00Z", "ERROR",    "payment-gateway", "Startup probe failed — service not ready after 30s"),
            ("2024-01-15T12:03:00Z", "CRITICAL", "api-gateway",     "Downstream payment-gateway unavailable — returning 503"),
            ("2024-01-15T12:05:00Z", "ERROR",    "postgres",        "Max client connections reached (100/100) — new connections refused"),
            ("2024-01-15T12:06:00Z", "WARN",     "ops-team",        "Manual restart of payment-gateway failed — root cause unresolved"),
        ]),
        description=(
            "payment-gateway is in CrashLoopBackOff. Each restart fails because "
            "the PostgreSQL connection pool is exhausted (100/100 connections). "
            "A service restart is needed to clear zombie connections, BUT the database "
            "team must also be escalated to reconfigure the pool limit. "
            "Both remediation AND escalation are required for full resolution."
        ),
    ),
    correct_classification=IncidentType.SERVICE_CRASH.value,
    correct_diagnosis="db_connection_pool_exhaustion",
    correct_remediation=ActionType.RESTART_SERVICE,
    requires_escalation=True,          # escalation is also required
    max_steps=12,
    reward_weights={
        "classification": 0.15,
        "diagnosis":      0.20,
        "remediation":    0.25,
        "escalation":     0.15,        # extra weight for escalation
        "resolution":     0.25,
    },
)


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskSpec] = {
    TASK_EASY.task_id:   TASK_EASY,
    TASK_MEDIUM.task_id: TASK_MEDIUM,
    TASK_HARD.task_id:   TASK_HARD,
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())
