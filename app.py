from fastapi import FastAPI, HTTPException
import uuid

from environment import CloudResourceEnv

app = FastAPI()

sessions = {}


# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------- RESET ----------------
@app.post("/reset")
def reset_env(req: dict = {}):
    try:
        task_level = req.get("task_level", "easy").upper()

        session_id = str(uuid.uuid4())

        env = CloudResourceEnv(task_level=task_level)
        obs = env.reset()

        sessions[session_id] = env

        return {
            "session_id": session_id,
            "observation": obs
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------- STEP ----------------
@app.post("/step")
def step_env(req: dict):
    try:
        session_id = req.get("session_id")
        action = req.get("action", {}).get("action_type", "no_op")

        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        env = sessions[session_id]

        obs, reward, done, info = env.step(action)

        return {
            "observation": obs,
            "reward": float(reward),
            "done": bool(done),
            "info": info if info else {}
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------- STATE ----------------
@app.get("/state/{session_id}")
def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    env = sessions[session_id]

    return {
        "session_id": session_id,
        "observation": env._get_obs(),
        "total_reward": float(env.total_reward),
        "done": env.time_step >= env.max_steps,
        "info": {}
    }
