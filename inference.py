"""
Baseline inference runner for Supply Chain OpenEnv.

The script emits only [START], [STEP], and [END] records on stdout.
All auxiliary diagnostics go to stderr.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import httpx
from openai import OpenAI

from graders import easy_grade, hard_grade, medium_grade
from graders.common import clamp

API_BASE_URL = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TEMPERATURE = 0.0
PROFILE_MAX_TOKENS = 16
SUCCESS_THRESHOLD = 0.6
UNIT_COST = 10.0

TASK_CONFIG = {
    "easy": {
        "max_steps": 10,
        "budget": 10000.0,
        "seed": 101,
        "default_profile": "BALANCED",
        "grader": easy_grade,
    },
    "medium": {
        "max_steps": 20,
        "budget": 18000.0,
        "seed": 202,
        "default_profile": "BALANCED",
        "grader": medium_grade,
    },
    "hard": {
        "max_steps": 30,
        "budget": 24000.0,
        "seed": 303,
        "default_profile": "CONSTRAINED",
        "grader": hard_grade,
    },
}

PROFILE_LIBRARY = {
    "LEAN": {
        "reorder_buffer_days": 1,
        "target_buffer_days": 4,
        "spike_factor": 1.00,
        "max_utilization": 0.62,
        "reserve_ratio": 0.05,
        "max_orders_per_step": 2,
        "emergency_reserve_release": 0.15,
    },
    "BALANCED": {
        "reorder_buffer_days": 2,
        "target_buffer_days": 5,
        "spike_factor": 1.10,
        "max_utilization": 0.70,
        "reserve_ratio": 0.15,
        "max_orders_per_step": 4,
        "emergency_reserve_release": 0.25,
    },
    "RESILIENT": {
        "reorder_buffer_days": 3,
        "target_buffer_days": 6,
        "spike_factor": 1.20,
        "max_utilization": 0.74,
        "reserve_ratio": 0.20,
        "max_orders_per_step": 5,
        "emergency_reserve_release": 0.30,
    },
    "CONSTRAINED": {
        "reorder_buffer_days": 2,
        "target_buffer_days": 5,
        "spike_factor": 1.25,
        "max_utilization": 0.68,
        "reserve_ratio": 0.35,
        "max_orders_per_step": 3,
        "emergency_reserve_release": 0.20,
    },
}


STRICT_EPSILON = 0.01


def stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def emit_log(tag: str, payload: Dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'))}", flush=True)


def log_start(task: str, env: str, model: str) -> None:
    emit_log("START", {"task": task, "env": env, "model": model})


def log_step(step: int, action: Any, reward: float, done: bool, error: str | None) -> None:
    emit_log(
        "STEP",
        {
            "step": step,
            "action": action,
            "reward": clamp(reward),
            "done": done,
            "error": error,
        },
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    emit_log(
        "END",
        {
            "success": success,
            "steps": steps,
            "score": clamp(score),
            "rewards": [clamp(reward) for reward in rewards],
        },
    )


class SupplyChainEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def reset(self, difficulty: str, seed: int) -> Dict[str, Any]:
        response = await self._client.post(
            f"{self.base_url}/reset",
            json={"difficulty": difficulty, "seed": seed},
        )
        response.raise_for_status()
        return response.json()

    async def step(self, episode_id: str, reorder_quantities: List[Dict[str, Any]]) -> Dict[str, Any]:
        response = await self._client.post(
            f"{self.base_url}/step",
            json={
                "episode_id": episode_id,
                "reorder_quantities": reorder_quantities,
            },
        )
        response.raise_for_status()
        return response.json()

    async def get_state(self, episode_id: str) -> Dict[str, Any]:
        response = await self._client.get(f"{self.base_url}/state/{episode_id}")
        response.raise_for_status()
        return response.json()

    async def health(self) -> int:
        response = await self._client.get(f"{self.base_url}/health")
        return response.status_code

    async def close(self) -> None:
        await self._client.aclose()


def summarise_inventory(observation: Dict[str, Any]) -> str:
    lines = []
    for sku in observation["skus"]:
        lines.append(
            (
                f"{sku['sku_id']}: stock={sku['stock']}, pending={sku['pending_order']}, "
                f"demand={sku['demand_per_day']}, lead={sku['supplier_delay_days']}, "
                f"reorder_point={sku['reorder_point']}"
            )
        )
    return "\n".join(lines)


def choose_profile(client: OpenAI, difficulty: str, observation: Dict[str, Any]) -> str:
    default_profile = TASK_CONFIG[difficulty]["default_profile"]
    prompt = (
        "Choose exactly one inventory control profile for this episode.\n"
        "Reply with one token only from this set: LEAN, BALANCED, RESILIENT, CONSTRAINED.\n\n"
        f"Difficulty: {difficulty}\n"
        f"Budget remaining: {observation['budget_remaining']}\n"
        f"Day: {observation['day']}\n"
        "Current SKUs:\n"
        f"{summarise_inventory(observation)}\n\n"
        "Guidance:\n"
        "- LEAN uses little inventory.\n"
        "- BALANCED fits stable multi-SKU tasks.\n"
        "- RESILIENT buys extra buffer for volatility.\n"
        "- CONSTRAINED protects budget and prioritises urgent items.\n"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Respond with one token only."},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=PROFILE_MAX_TOKENS,
            stream=False,
        )
        token = (completion.choices[0].message.content or "").strip().upper()
        token = token.replace('"', "").replace("'", "").split()[0].strip(".,!;:")
        if token in PROFILE_LIBRARY:
            return token
    except Exception as exc:
        stderr(f"[PROFILE_ERROR] {difficulty}: {exc}")

    return default_profile


def _candidate_order(
    sku: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any] | None:
    demand = max(1, int(sku["demand_per_day"]))
    lead = max(1, int(sku["supplier_delay_days"]))
    stock = int(sku["stock"])
    pending = int(sku["pending_order"])
    max_stock = int(sku["max_stock"])
    inventory_position = stock + pending

    if pending > 0:
        return None

    reorder_window = lead + profile["reorder_buffer_days"]
    target_window = lead + profile["target_buffer_days"]
    days_cover = inventory_position / demand
    cap_stock = int(max_stock * profile["max_utilization"])
    target_stock = min(cap_stock, int(round(demand * target_window * profile["spike_factor"])))
    target_stock = max(target_stock, int(sku["reorder_point"]))

    if inventory_position >= target_stock:
        return None

    if days_cover > reorder_window + 0.5 and stock > int(sku["reorder_point"]):
        return None

    needed = min(max_stock - inventory_position, target_stock - inventory_position)
    if needed <= 0:
        return None

    urgency = max(0.0, reorder_window - days_cover) * demand
    urgency += max(0, int(sku["reorder_point"]) - stock)
    urgency += demand * (1.5 if stock <= demand * lead else 0.0)
    min_batch = max(demand, int(round(demand * min(lead, 3) * 0.75)))
    quantity = max(min_batch, int(needed))

    return {
        "sku_id": sku["sku_id"],
        "quantity": quantity,
        "urgency": urgency,
        "demand": demand,
        "days_cover": round(days_cover, 4),
    }


def build_action(observation: Dict[str, Any], difficulty: str, profile_name: str) -> List[Dict[str, Any]]:
    profile = PROFILE_LIBRARY[profile_name]
    task_cfg = TASK_CONFIG[difficulty]
    budget_remaining = float(observation["budget_remaining"])
    budget_units = int(budget_remaining // UNIT_COST)
    if budget_units <= 0:
        return []

    day = int(observation["day"])
    days_remaining = max(1, task_cfg["max_steps"] - day)
    reserve_budget = task_cfg["budget"] * profile["reserve_ratio"] * (days_remaining / task_cfg["max_steps"])
    spendable_budget = max(0.0, budget_remaining - reserve_budget)
    spendable_units = int(spendable_budget // UNIT_COST)

    candidates = []
    for sku in observation["skus"]:
        candidate = _candidate_order(sku, profile)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: (item["urgency"], item["demand"], -item["days_cover"]),
        reverse=True,
    )

    if not candidates:
        return []

    if spendable_units <= 0:
        critical_candidate = candidates[0]
        spendable_units = min(
            budget_units,
            max(
                critical_candidate["demand"],
                int(budget_units * profile["emergency_reserve_release"]),
            ),
        )

    actions: List[Dict[str, Any]] = []
    remaining_units = spendable_units

    for candidate in candidates[: profile["max_orders_per_step"]]:
        if remaining_units <= 0:
            break

        quantity = min(candidate["quantity"], remaining_units)
        quantity = max(candidate["demand"], int(quantity))
        quantity = min(quantity, remaining_units)
        if quantity < candidate["demand"]:
            continue

        actions.append({"sku_id": candidate["sku_id"], "quantity": int(quantity)})
        remaining_units -= int(quantity)

    return actions


async def run_task(difficulty: str, llm_client: OpenAI, env_client: SupplyChainEnvClient) -> float:
    cfg = TASK_CONFIG[difficulty]
    task_name = f"supply_chain_{difficulty}"
    benchmark = "supply-chain-openenv"
    rewards: List[float] = []
    steps_taken = 0
    score = STRICT_EPSILON
    success = False

    log_start(task=task_name, env=benchmark, model=MODEL_NAME)

    result = await env_client.reset(difficulty, cfg["seed"])
    episode_id = result["episode_id"]
    observation = result["observation"]
    profile_name = choose_profile(llm_client, difficulty, observation)

    try:
        for step in range(1, cfg["max_steps"] + 1):
            if result.get("done", False):
                break

            action = build_action(observation, difficulty, profile_name)
            result = await env_client.step(episode_id, action)
            observation = result["observation"]
            reward_payload = result.get("reward", {})
            reward = (
                float(reward_payload.get("value", 0.0))
                if isinstance(reward_payload, dict)
                else float(reward_payload or 0.0)
            )
            reward = clamp(reward)
            done = bool(result.get("done", False))

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=None)

            if done:
                break

        final_state = await env_client.get_state(episode_id)
        score = clamp(float(cfg["grader"](final_state, None, None)))
        success = score >= SUCCESS_THRESHOLD
    except Exception as exc:
        stderr(f"[TASK_ERROR] {difficulty}: {exc}")
        log_step(
            step=steps_taken + 1,
            action=[],
            reward=STRICT_EPSILON,
            done=True,
            error=str(exc),
        )
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY or HF_TOKEN environment variable is required.")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = SupplyChainEnvClient(ENV_BASE_URL)

    try:
        for _ in range(20):
            try:
                if await env_client.health() == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(3)
        else:
            raise RuntimeError("Environment server did not become ready in time.")

        for difficulty in ("easy", "medium", "hard"):
            await run_task(difficulty, llm_client, env_client)
    finally:
        await env_client.close()


if __name__ == "__main__":
    asyncio.run(main())
