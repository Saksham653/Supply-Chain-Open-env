---
title: Supply Chain OpenEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - supply-chain
  - warehouse
  - inventory
---

# Supply Chain OpenEnv

Supply Chain OpenEnv is a real-world warehouse inventory benchmark for agents.
Each episode simulates the job of an inventory planner deciding when to reorder
stock, how much to buy, and how to preserve budget while serving demand.

## Why this environment

Warehouse replenishment is a real business workflow, not a toy problem. Teams
make these decisions every day under uncertainty: demand changes, suppliers
slip, budgets are finite, and both stockouts and over-ordering are expensive.

This benchmark turns that workflow into a reproducible OpenEnv environment with
seeded episodes, dense rewards, deterministic graders, and a container-friendly
API.

## Project structure

```text
Supply-Chain-Open-env/
|-- app.py
|-- inference.py
|-- openenv.yaml
|-- Dockerfile
|-- requirements.txt
|-- environment/
|   |-- __init__.py
|   |-- env.py
|   |-- models.py
|   `-- simulator.py
`-- graders/
    |-- __init__.py
    |-- common.py
    |-- easy.py
    |-- medium.py
    `-- hard.py
```

## Environment design

Episodes run in daily timesteps:

1. Pending deliveries arrive.
2. The agent submits a reorder action.
3. Daily demand is applied to each SKU.
4. The environment returns the next observation and a shaped reward.

The state returned by `/state/{episode_id}` includes both the live warehouse
snapshot and the full trajectory so far, which is what the graders score.

## Observation space

Each `reset()` and `step()` response returns:

| Field | Type | Meaning |
| --- | --- | --- |
| `skus` | array | Current per-SKU warehouse state |
| `day` | int | Current day index |
| `budget_remaining` | float | Remaining purchasing budget |
| `stockouts_today` | int | Number of SKUs that stocked out today |
| `reward` | float | Dense per-step reward in `[0.0, 1.0]` |
| `done` | bool | Whether the episode has ended |
| `message` | string | Human-readable daily summary |

Each SKU record contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `sku_id` | string | SKU identifier such as `SKU_A` |
| `stock` | int | Units currently on hand |
| `demand_per_day` | int | Base daily demand |
| `reorder_point` | int | Suggested reorder trigger |
| `max_stock` | int | Capacity ceiling for the SKU |
| `supplier_delay_days` | int | Base supplier lead time |
| `pending_order` | int | Units currently in transit |
| `days_until_arrival` | int | Days until the pending order arrives |

## Action space

The agent sends a JSON object to `/step`:

```json
{
  "episode_id": "uuid-from-reset",
  "reorder_quantities": [
    {"sku_id": "SKU_A", "quantity": 30},
    {"sku_id": "SKU_C", "quantity": 20}
  ]
}
```

Action rules:

- An empty `reorder_quantities` array is a valid no-op.
- Each unit costs `$10`.
- Orders are rejected when budget is insufficient.
- Only one pending order per SKU can exist at once.
- Quantities must be positive integers.

## Reward shaping

The reward is dense and intentionally non-binary:

```text
reward = max(
  0.0,
  1.0
  - 0.65 * service_penalty
  - 0.20 * overstock_penalty
  - 0.15 * critical_inventory_penalty
  - 0.10 * rejection_penalty
)
```

Where:

- `service_penalty = units_unfulfilled / units_demanded`
- `overstock_penalty = overstocked_skus / num_skus`
- `critical_inventory_penalty = critical_skus / num_skus`
- `rejection_penalty = rejected_orders / num_skus`

This gives meaningful partial credit throughout the trajectory and penalises
wasteful or clearly unhelpful ordering behavior.

## Tasks

### Easy: Single SKU Reorder

- Horizon: 10 days
- SKUs: 1
- Budget: `$10,000`
- Dynamics: deterministic demand and deterministic lead time
- Default seed: `101`
- Grader: `graders/easy.py`

Objective: reorder one predictable item before it stocks out.

### Medium: Multi-SKU Inventory Balance

- Horizon: 20 days
- SKUs: 10
- Budget: `$18,000`
- Dynamics: plus or minus 20 percent demand variation, plus or minus 1 day lead-time variation
- Default seed: `202`
- Grader: `graders/medium.py`

Objective: keep service level high without drifting into persistent overstock.

### Hard: Constrained Supply Chain

- Horizon: 30 days
- SKUs: 10
- Budget: `$24,000`
- Dynamics: demand spikes up to 2.5x, random supplier delays, tighter budget
- Default seed: `303`
- Grader: `graders/hard.py`

Objective: prioritise scarce budget under volatility and keep late-episode
performance from collapsing.

## Graders

Each grader returns a deterministic score in `[0.0, 1.0]` from the exported
episode state:

- `easy.py`: prioritises zero stockouts on the single-SKU task.
- `medium.py`: balances service level, stockout-free days, overstock control, and rejected orders.
- `hard.py`: rewards service level and late-episode resilience under constraint.

## Baseline inference

The root `inference.py` script:

- uses the OpenAI client
- reads credentials from `OPENAI_API_KEY` or `HF_TOKEN`
- resets each task with a fixed default seed
- emits only `[START]`, `[STEP]`, and `[END]` records on stdout
- uses a hybrid baseline where the model selects a strategy profile and a deterministic allocator executes the daily orders

### Baseline score notes

The benchmark now uses fixed seeds (`101`, `202`, `303`) so scores are
reproducible across runs with the same model and endpoint.

Latest seeded baseline from `inference.py` with `openai/gpt-4o-mini`:

| Task | Seed | Score | Pass |
| --- | --- | --- | --- |
| Easy | `101` | `1.0000` | Yes |
| Medium | `202` | `1.0000` | Yes |
| Hard | `303` | `0.6661` | Yes |

Average seeded score: `0.8887`

## Setup

### Local run

Start the environment server:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

In a second terminal:

```bash
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="openai/gpt-4o-mini"
export HF_TOKEN="your_api_key_here"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

If you are using a direct OpenAI key instead of OpenRouter:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your_openai_key_here"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

### Docker

```bash
docker build -t supply-chain-openenv .
docker run -p 7860:7860 supply-chain-openenv
```

Then run inference against the live container from another shell.

## Manual API usage

```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty":"medium","seed":202}'
```

## Validation

```bash
pip install openenv-core
openenv validate openenv.yaml
```

## Hugging Face Spaces

This repo is set up for a Docker Space. Use the Space secrets UI to provide:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`

The app listens on port `7860`, which matches the Docker Space configuration.
