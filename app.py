import asyncio
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException

load_dotenv()

from environment.env import (
    DEFAULT_SEEDS,
    SupplyChainAction,
    SupplyChainEnvironment,
)
from graders.common import clamp
from environment.models import (
    EpisodeStateModel,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    SupplyChainObservationModel,
    TaskListResponse,
    TaskModel,
)
from inference import (
    API_BASE_URL,
    API_KEY,
    ENV_BASE_URL,
    SupplyChainEnvClient,
    run_task,
)
from openai import OpenAI

app = FastAPI(title="Supply Chain OpenEnv", version="1.1.0")

_envs: Dict[str, SupplyChainEnvironment] = {}


def _build_response(episode_id: str, observation, state_dict: dict, done: bool) -> dict:
    return {
        "episode_id": episode_id,
        "observation": SupplyChainObservationModel(**observation.__dict__),
        "reward": clamp(float(observation.reward or 0.0)),
        "done": done,
        "state": EpisodeStateModel(**state_dict),
    }


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest | None = Body(default=None)) -> ResetResponse:
    req = req or ResetRequest()
    env = SupplyChainEnvironment(difficulty=req.difficulty, seed=req.seed)
    observation = env.reset()
    episode_id = env.state.episode_id
    _envs[episode_id] = env
    payload = _build_response(episode_id, observation, env.export_state(), False)
    return ResetResponse(**payload)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    env = _envs.get(req.episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found. Call /reset first.")

    action = SupplyChainAction(
        reorder_quantities=[item.model_dump() for item in req.reorder_quantities]
    )
    observation = env.step(action)
    payload = _build_response(req.episode_id, observation, env.export_state(), observation.done)
    return StepResponse(**payload)


@app.get("/state/{episode_id}", response_model=EpisodeStateModel)
def state(episode_id: str) -> EpisodeStateModel:
    env = _envs.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found.")
    return EpisodeStateModel(**env.export_state())


@app.get("/tasks", response_model=TaskListResponse)
def list_tasks() -> TaskListResponse:
    return TaskListResponse(
        tasks=[
            TaskModel(
                id="easy",
                name="Single SKU Reorder",
                description="Manage one product and reorder before stockout over a short horizon.",
                difficulty="easy",
                episode_length=10,
                success_threshold=0.6,
                default_seed=DEFAULT_SEEDS["easy"],
            ),
            TaskModel(
                id="medium",
                name="Multi-SKU Inventory Balance",
                description="Manage ten SKUs while balancing service level against overstock.",
                difficulty="medium",
                episode_length=20,
                success_threshold=0.6,
                default_seed=DEFAULT_SEEDS["medium"],
            ),
            TaskModel(
                id="hard",
                name="Constrained Supply Chain",
                description="Handle demand spikes, supplier delays, and a tighter budget across ten SKUs.",
                difficulty="hard",
                episode_length=30,
                success_threshold=0.6,
                default_seed=DEFAULT_SEEDS["hard"],
            ),
        ]
    )


def run_agent(state: dict) -> dict:
    """Thin wrapper used by the /run endpoint."""
    difficulty = state.get("difficulty", "easy")
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = SupplyChainEnvClient(ENV_BASE_URL)
    try:
        score = asyncio.run(run_task(difficulty, llm_client, env_client))
    finally:
        asyncio.run(env_client.close())
    return {"score": score}


@app.post("/run")
def run():
    results = {}

    for difficulty in ["easy", "medium", "hard"]:
        env = SupplyChainEnvironment(difficulty=difficulty)
        obs = env.reset()

        # episode_id lives on the environment state, not the observation
        episode_id = env.state.episode_id

        total_reward = 0.0
        steps = 5

        for _ in range(steps):
            # env.step() expects a SupplyChainAction dataclass (kw_only)
            action = SupplyChainAction(reorder_quantities=[])
            obs = env.step(action)

            # obs is a SupplyChainObservation dataclass — use attribute access
            reward = float(obs.reward) if hasattr(obs, "reward") else float(obs.get("reward", 0.0))
            total_reward += clamp(reward)

        avg_reward = total_reward / steps
        results[difficulty] = clamp(avg_reward)

    return {
    "tasks": [
        {"task": "easy", "score": results["easy"]},
        {"task": "medium", "score": results["medium"]},
        {"task": "hard", "score": results["hard"]}
    ]
}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
