from __future__ import annotations

from env import IncidentResponseEnv
from models import Action, Observation
from tasks import TASK_REGISTRY, list_tasks


class RuleBasedAgent:
    def __init__(self):
        self.history = []

    def reset(self):
        self.history = []

    def act(self, obs: Observation) -> Action:
        phase = str(obs.phase).lower()

        if "verify" in phase:
            return Action(action_type="escalate", parameters={})

        if "detect" in phase or "class" in phase:
            return Action(
                action_type="classify",
                parameters={"incident_type": self._guess_incident(obs)}
            )

        if "diag" in phase:
            return Action(
                action_type="diagnose",
                parameters={"root_cause": self._diagnose(obs)}
            )

        if "remed" in phase:
            return Action(
                action_type=self._remediate(obs),
                parameters={}
            )

        return Action(action_type="escalate", parameters={})

    def _guess_incident(self, obs):
        text = getattr(obs, "incident_description", "").lower()

        if "cpu" in text:
            return "cpu_high"
        if "disk" in text:
            return "disk_full"
        if "crash" in text:
            return "service_crash"

        return "service_crash"

    def _diagnose(self, obs):
        incident = getattr(obs, "classification_result", "")

        if incident == "cpu_high":
            return "runaway_nginx_worker"
        if incident == "disk_full":
            return "unrotated_log_accumulation"
        if incident == "service_crash":
            return "db_connection_pool_exhaustion"

        return "unknown"

    def _remediate(self, obs):
        incident = getattr(obs, "classification_result", "")

        if incident == "cpu_high":
            return "restart_service"
        if incident == "disk_full":
            return "clear_logs"
        if incident == "service_crash":
            return "restart_service"

        return "restart_service"


def main():
    env = IncidentResponseEnv()
    agent = RuleBasedAgent()

    for task_id in list_tasks():

        if task_id not in TASK_REGISTRY:
            continue

        print(f"[START] task={task_id}", flush=True)

        obs = env.reset(task_id)
        agent.reset()

        done = False
        step = 0

        while not done:
            step += 1
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)

            reward_value = reward.total if hasattr(reward, "total") else reward

            print(
                f"[STEP] step={step} reward={reward_value}",
                flush=True
            )

        result = env.compute_final_score()

        print(
            f"[END] task={task_id} score={result.score} steps={step}",
            flush=True
        )


if __name__ == "__main__":
    main()
