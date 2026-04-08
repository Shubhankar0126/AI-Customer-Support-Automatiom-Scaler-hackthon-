from fastapi import FastAPI
from env import IncidentResponseEnv
from models import Action

app = FastAPI()
env = IncidentResponseEnv()

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/reset")
def reset(task_id: str = "task_easy_cpu_high"):
    obs = env.reset(task_id)
    return obs.model_dump()

@app.post("/step")
def step(action: dict):
    obs, reward, done, info = env.step(Action(**action))
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state().model_dump()
