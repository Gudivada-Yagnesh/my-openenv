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
@app.post("/reset", response_model=ResetResponse)
def reset_env(req: ResetRequest):
    session_id = req.session_id or str(uuid.uuid4())

    env = CloudResourceEnv(task=req.task_level)
    obs = env.reset()

    sessions[session_id] = env

    return ResetResponse(
        session_id=session_id,
        observation=Observation(**obs)
    )


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