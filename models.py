from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal


# ---------------- OBSERVATION ----------------
class Observation(BaseModel):
    cpu: float
    memory: float
    predicted_memory: float
    queue: int
    workload: float
    compression: bool
    step: int


# ---------------- ACTION ----------------
class Action(BaseModel):
    action_type: Literal[
        "scale_up_resources",
        "scale_down_resources",
        "compress_memory",
        "schedule_tasks",
        "no_op"
    ]


# ---------------- STATE ----------------
class State(BaseModel):
    session_id: str
    observation: Observation
    total_reward: float
    done: bool
    info: Dict[str, Any]


# ---------------- STEP ----------------
class StepRequest(BaseModel):
    session_id: str
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------- RESET ----------------
class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_level: Literal["easy", "medium", "hard"] = "easy"


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation