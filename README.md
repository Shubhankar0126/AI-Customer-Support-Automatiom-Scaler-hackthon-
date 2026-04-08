# AI Incident Response & IT Ops Automation — OpenEnv

**Team:** namoBharat
**Author:** Shubhankar Pandey
**Team Member:** Meera Singh
**Hackathon:** Meta × PyTorch OpenEnv Challenge

---

# Environment Description

This project implements a **production-ready OpenEnv environment** that simulates real-world IT incident response workflows.
An AI agent must:

* classify incidents
* diagnose root causes
* apply remediation
* escalate when necessary
* verify resolution

The workflow follows:

```
DETECTION → CLASSIFICATION → DIAGNOSIS → REMEDIATION → VERIFICATION → CLOSED
```

Each phase restricts valid actions and provides structured observations.

---

# Incident Types

| Type          | Description                             |
| ------------- | --------------------------------------- |
| cpu_high      | High CPU usage due to runaway process   |
| disk_full     | Disk filled by unrotated logs           |
| service_crash | Service crash due to DB pool exhaustion |

---

# Observation Space

Each `step()` returns an Observation object:

* incident_id
* phase
* step_number
* max_steps
* metrics
* recent_alerts
* incident_description
* classified
* classification_result
* diagnosed
* diagnosis_result
* remediation_applied
* remediation_action
* resolved
* escalated
* allowed_actions
* status_message

---

# Action Space

| Action          | Description                |
| --------------- | -------------------------- |
| classify        | classify incident type     |
| diagnose        | provide root cause         |
| restart_service | restart affected service   |
| clear_logs      | free disk space            |
| scale_up        | increase compute resources |
| escalate        | escalate to senior team    |

---

# Reward Shaping

| Event                  | Reward         |
| ---------------------- | -------------- |
| Correct classification | +0.20          |
| Correct diagnosis      | +0.25          |
| Correct remediation    | +0.30          |
| Incident resolved      | +0.25          |
| Correct escalation     | +0.15          |
| Wrong action           | -0.10          |
| Wrong remediation      | -0.05          |
| Step penalty           | -0.02 to -0.05 |

Final score normalized to **0.0 – 1.0**

---

# Tasks

## Easy — CPU High

```
classify(cpu_high)
→ diagnose(runaway_nginx_worker)
→ restart_service
```

## Medium — Disk Full

```
classify(disk_full)
→ diagnose(unrotated_log_accumulation)
→ clear_logs
```

## Hard — Service Crash

```
classify(service_crash)
→ diagnose(db_connection_pool_exhaustion)
→ restart_service
→ escalate
```

---

# Setup Instructions

## Install dependencies

```
pip install -r requirements.txt
```

## Run baseline

```
python inference.py
```

## Run specific task

```
python inference.py --task task_easy_cpu_high
```

---

# Expected Output

```
TASK                                 SCORE
-------------------------------------------
task_easy_cpu_high                  0.89
task_medium_disk_full               0.90
task_hard_service_crash             0.87
-------------------------------------------
AVERAGE                             ~0.89
```

---

# Docker

Build container:

```
docker build -t ai-incident-env .
```

Run:

```
docker run ai-incident-env
```

---

# Project Structure

```
ai_incident_env/
├── models.py
├── tasks.py
├── graders.py
├── env.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# API Usage

```
from env import IncidentResponseEnv
from models import Action

env = IncidentResponseEnv()
obs = env.reset("task_easy_cpu_high")

obs, reward, done, _ = env.step(
    Action(action_type="classify", parameters={"incident_type": "cpu_high"})
)
```

---

# Reproducibility

* deterministic rule-based agent
* fixed reward shaping
* no randomness in baseline

---

# Team

**Team Name:** namoBharat
**Members:**

* Shubhankar Pandey
* Meera Singh

---

# License

MIT
