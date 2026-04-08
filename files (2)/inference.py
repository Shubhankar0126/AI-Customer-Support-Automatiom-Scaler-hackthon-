from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

from env import IncidentResponseEnv
from models import Action, Observation
from tasks import TASK_REGISTRY, list_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("inference")


class RuleBasedAgent:
    def __init__(self):
        self.history = []

    def reset(self):
        self.history = []

    def act(self, obs: Observation) -> Action:
        phase = str(obs.phase).lower()

        if "verify" in phase:
            return Action(
            action_type="escalate",
            parameters={}
        )

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

        return Action(
        action_type="escalate",
        parameters={}
    )
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


def run_episode(
    env: IncidentResponseEnv,
    agent: RuleBasedAgent,
    task_id: str,
    verbose: bool = True,
):
    obs = env.reset(task_id)
    agent.reset()

    done = False
    total_reward = 0.0

    if verbose:
        logger.info("=" * 60)
        logger.info("TASK: %s", task_id)

    while not done:
        action = agent.act(obs)

        if verbose:
            logger.info(
                "Action: %-20s Params: %s",
                action.action_type,
                action.parameters
            )

        obs, reward, done, info = env.step(action)

        total_reward += reward.total if hasattr(reward, "total") else reward

        if verbose:
            logger.info("Reward: %s", reward)

    result = env.compute_final_score()

    if verbose:
        logger.info("FINAL SCORE: %.4f", result.score)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    verbose = not args.quiet

    env = IncidentResponseEnv()
    agent = RuleBasedAgent()

    task_ids = [args.task] if args.task else list_tasks()

    all_results = []

    for task_id in task_ids:
        if task_id not in TASK_REGISTRY:
            logger.error("Unknown task %s", task_id)
            continue

        result = run_episode(env, agent, task_id, verbose)
        all_results.append(result.model_dump())

    print("\n" + "=" * 60)
    print(f"{'TASK':<35} {'SCORE':>6}")
    print("-" * 60)

    for r in all_results:
        print(f"{r['task_id']:<35} {r['score']:>6.4f}")

    avg = sum(r["score"] for r in all_results) / len(all_results)

    print("-" * 60)
    print(f"{'AVERAGE':<35} {avg:>6.4f}")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()