from fastapi import FastAPI, HTTPException
import uuid

from models import (
    StepRequest,
    StepResponse,
    ResetRequest,
    ResetResponse,
    State,
    Observation
)

from server.environment import CloudResourceEnv

app = FastAPI()

# ---------------- SESSION STORE ----------------
sessions = {}

# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------- RESET ----------------
from fastapi import FastAPI
import uuid

app = FastAPI()

sessions = {}

@app.post("/reset")
def reset_env(req: dict = {}):
    task_level = req.get("task_level", "EASY")

    session_id = str(uuid.uuid4())

    from environment import CloudResourceEnv
    env = CloudResourceEnv(task_level=task_level)

    obs = env.reset()

    sessions[session_id] = env

    return {
        "session_id": session_id,
        "observation": obs
    }

# ---------------- STEP ----------------
@app.post("/step", response_model=StepResponse)
def step_env(req: StepRequest):

    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    env = sessions[req.session_id]

    obs, reward, done, info = env.step(req.action.action_type)

    return StepResponse(
        observation=Observation(**obs),
        reward=reward,
        done=done,
        info=info
    )

@app.post("/step")
def step_env(req: dict):
    session_id = req.get("session_id")
    action = req.get("action", {}).get("action_type")

    if session_id not in sessions:
        return {"error": "Session not found"}

    env = sessions[session_id]

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }
# ---------------- STATE ----------------
@app.get("/state/{session_id}", response_model=State)
def get_state(session_id: str):

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    env = sessions[session_id]

    return State(
        session_id=session_id,
        observation=Observation(**env._obs()),
        total_reward=env.total_reward,
        done=(env.time >= env.max_steps),
        info={
            "score": env._score()
        }
    )
