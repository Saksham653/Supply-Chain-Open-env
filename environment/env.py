import random
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openenv_core.env_server import Action, Environment, Observation, State

from environment.simulator import (
    DailyStepRunner,
    DemandSimulator,
    ReorderProcessor,
    SKU,
    SKUFactory,
    SupplierSimulator,
)
from graders import easy_grade, medium_grade, hard_grade
from graders.common import clamp, extract_metrics


@dataclass(kw_only=True)
class SupplyChainAction(Action):
    reorder_quantities: List[dict]


@dataclass(kw_only=True)
class SupplyChainObservation(Observation):
    skus: List[dict]
    day: int
    budget_remaining: float
    stockouts_today: int
    reward: float
    done: bool
    message: str


@dataclass
class SupplyChainState(State):
    episode_id: str = ""
    step_count: int = 0
    difficulty: str = "easy"
    seed: int = 0
    day: int = 0
    max_days: int = 0
    budget_start: float = 0.0
    budget_remaining: float = 0.0
    total_reward: float = 0.0
    total_stockouts: int = 0
    stockout_days: int = 0
    total_units_demanded: int = 0
    total_units_unfulfilled: int = 0
    total_orders_requested: int = 0
    total_orders_placed: int = 0
    total_orders_rejected: int = 0
    done: bool = False


DAYS_PER_DIFFICULTY = {
    "easy": 10,
    "medium": 20,
    "hard": 30,
}

BUDGET_PER_DIFFICULTY = {
    "easy": 10000.0,
    "medium": 18000.0,
    "hard": 24000.0,
}

DEFAULT_SEEDS = {
    "easy": 101,
    "medium": 202,
    "hard": 303,
}

STRICT_EPSILON = 0.01


class SupplyChainEnvironment(Environment):
    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None):
        super().__init__()
        assert difficulty in ("easy", "medium", "hard"), f"Invalid difficulty: {difficulty}"
        self.difficulty = difficulty
        self.seed = DEFAULT_SEEDS[difficulty] if seed is None else seed

        self._rng = random.Random(self.seed)
        self._demand_sim = DemandSimulator(difficulty, self._rng)
        self._supplier_sim = SupplierSimulator(difficulty, self._rng)
        self._reorder_proc = ReorderProcessor(self._supplier_sim)
        self._daily_runner = DailyStepRunner(self._demand_sim)

        self._skus: List[SKU] = []
        self._initial_skus: List[dict] = []
        self._trajectory: List[Dict[str, Any]] = []
        self._day = 0
        self._budget = 0.0
        self._starting_budget = 0.0
        self._max_days = 0
        self._total_reward = 0.0
        self._total_stockouts = 0
        self._stockout_days = 0
        self._overstock_days = 0
        self._critical_days = 0
        self._total_units_demanded = 0
        self._total_units_unfulfilled = 0
        self._total_orders_requested = 0
        self._total_orders_placed = 0
        self._total_orders_rejected = 0
        self._state = SupplyChainState()

    def reset(self) -> SupplyChainObservation:
        self._rng = random.Random(self.seed)
        self._demand_sim = DemandSimulator(self.difficulty, self._rng)
        self._supplier_sim = SupplierSimulator(self.difficulty, self._rng)
        self._reorder_proc = ReorderProcessor(self._supplier_sim)
        self._daily_runner = DailyStepRunner(self._demand_sim)

        self._skus = SKUFactory.generate(self.difficulty, self._rng)
        self._initial_skus = self._serialize_skus()
        self._trajectory = []
        self._day = 0
        self._budget = BUDGET_PER_DIFFICULTY[self.difficulty]
        self._starting_budget = self._budget
        self._max_days = DAYS_PER_DIFFICULTY[self.difficulty]
        self._total_reward = 0.0
        self._total_stockouts = 0
        self._stockout_days = 0
        self._overstock_days = 0
        self._critical_days = 0
        self._total_units_demanded = 0
        self._total_units_unfulfilled = 0
        self._total_orders_requested = 0
        self._total_orders_placed = 0
        self._total_orders_rejected = 0
        self._state = SupplyChainState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            difficulty=self.difficulty,
        )
        self._refresh_state(done=False)

        return SupplyChainObservation(
            skus=self._serialize_skus(),
            day=self._day,
            budget_remaining=self._budget,
            stockouts_today=0,
            reward=clamp(0.01),
            done=False,
            message=(
                f"Episode started. Difficulty: {self.difficulty}. "
                f"Seed: {self.seed}. You have {self._max_days} days and "
                f"${self._budget:.0f} budget."
            ),
        )

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        if self._state.done:
            raise RuntimeError("Episode already completed. Call reset() for a new episode.")

        self._state.step_count += 1
        self._day += 1

        arrivals = self._reorder_proc.process_arrivals(self._skus)
        reorder_map = {
            item["sku_id"]: item["quantity"]
            for item in action.reorder_quantities
            if item.get("quantity", 0) > 0
        }
        self._total_orders_requested += len(reorder_map)

        budget_before_orders = self._budget
        self._budget, receipts = self._reorder_proc.place_orders(self._skus, reorder_map, self._budget)

        demand_metrics = self._daily_runner.run(self._skus)
        stockouts_today = int(demand_metrics["stockouts"])
        units_demanded = int(demand_metrics["units_demanded"])
        units_unfulfilled = int(demand_metrics["units_unfulfilled"])
        overstock_count = sum(1 for sku in self._skus if sku.stock > sku.max_stock * 0.8)
        critical_count = sum(1 for sku in self._skus if self._is_critical(sku))
        rejected_orders = sum(1 for receipt in receipts if receipt["status"] == "rejected")
        placed_orders = sum(1 for receipt in receipts if receipt["status"] == "placed")
        spend_today = round(budget_before_orders - self._budget, 2)

        # Build a state snapshot for the grader
        state_snapshot = self.export_state()
        
        if self.difficulty == "easy":
          reward = easy_grade(state_snapshot)    # now uses new signature
        elif self.difficulty == "medium":
          reward = medium_grade(state_snapshot)  # passes state, not pre-computed metrics
        else:
          reward = hard_grade(state_snapshot)
    
        reward = clamp(reward)

        done = state_snapshot["day"] >= state_snapshot["max_days"]

        self._total_reward += reward
        self._total_stockouts += stockouts_today
        self._stockout_days += int(stockouts_today > 0)
        self._overstock_days += int(overstock_count > 0)
        self._critical_days += int(critical_count > 0)
        self._total_units_demanded += units_demanded
        self._total_units_unfulfilled += units_unfulfilled
        self._total_orders_placed += placed_orders
        self._total_orders_rejected += rejected_orders
        self._trajectory.append(
            {
                "day": self._day,
                "reward": reward,
                "stockouts_today": stockouts_today,
                "units_demanded": units_demanded,
                "units_unfulfilled": units_unfulfilled,
                "service_level_today": clamp(
                    1.0 - (units_unfulfilled / units_demanded if units_demanded else 0.0)
                ),
                "budget_remaining": round(self._budget, 2),
                "spend_today": spend_today,
                "orders_requested": len(reorder_map),
                "orders_placed": placed_orders,
                "rejected_orders": rejected_orders,
                "overstocked_skus": overstock_count,
                "critical_skus": critical_count,
                "arrivals": arrivals,
                "receipts": receipts,
                "demand_details": demand_metrics["demand_details"],
            }
        )
        self._refresh_state(done=done)

        return SupplyChainObservation(
            skus=self._serialize_skus(),
            day=self._day,
            budget_remaining=self._budget,
            stockouts_today=stockouts_today,
            reward=reward,
            done=done,
            message=(
                f"Day {self._day}/{self._max_days}. "
                f"Stockouts: {stockouts_today}. "
                f"Unfulfilled units: {units_unfulfilled}. "
                f"Budget left: ${self._budget:.0f}."
            ),
        )

    @property
    def state(self) -> SupplyChainState:
        return self._state

    def export_state(self) -> Dict[str, Any]:
        average_reward = self._total_reward / self._day if self._day else 0.0
        total_reward = self._total_reward / self._max_days if self._max_days else 0.0
        service_level = 1.0 - (
            self._total_units_unfulfilled / self._total_units_demanded
            if self._total_units_demanded
            else 0.0
        )
        rejection_rate = (
            self._total_orders_rejected / self._total_orders_requested
            if self._total_orders_requested
            else 0.0
        )

        return {
            "episode_id": self._state.episode_id,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "day": self._day,
            "max_days": self._max_days,
            "done": self._state.done,
            "budget_start": self._starting_budget,
            "budget_remaining": round(self._budget, 2),
            "total_reward": clamp(total_reward),
            "average_reward": clamp(average_reward),
            "total_stockouts": self._total_stockouts,
            "stockout_days": self._stockout_days,
            "overstock_days": self._overstock_days,
            "critical_days": self._critical_days,
            "total_units_demanded": self._total_units_demanded,
            "total_units_unfulfilled": self._total_units_unfulfilled,
            "service_level": clamp(service_level),
            "total_orders_requested": self._total_orders_requested,
            "total_orders_placed": self._total_orders_placed,
            "total_orders_rejected": self._total_orders_rejected,
            "rejection_rate": clamp(rejection_rate),
            "initial_skus": deepcopy(self._initial_skus),
            "skus": self._serialize_skus(),
            "trajectory": deepcopy(self._trajectory),
        }

    def _refresh_state(self, done: bool) -> None:
        total_reward = self._total_reward / self._max_days if self._max_days else 0.0
        self._state.day = self._day
        self._state.max_days = self._max_days
        self._state.seed = self.seed
        self._state.budget_start = self._starting_budget
        self._state.budget_remaining = round(self._budget, 2)
        self._state.total_reward = clamp(total_reward)
        self._state.total_stockouts = self._total_stockouts
        self._state.stockout_days = self._stockout_days
        self._state.total_units_demanded = self._total_units_demanded
        self._state.total_units_unfulfilled = self._total_units_unfulfilled
        self._state.total_orders_requested = self._total_orders_requested
        self._state.total_orders_placed = self._total_orders_placed
        self._state.total_orders_rejected = self._total_orders_rejected
        self._state.done = done

    def _serialize_skus(self) -> List[dict]:
        return [
            {
                "sku_id": sku.sku_id,
                "stock": sku.stock,
                "demand_per_day": sku.demand_per_day,
                "reorder_point": sku.reorder_point,
                "max_stock": sku.max_stock,
                "supplier_delay_days": sku.supplier_delay_days,
                "pending_order": sku.pending_order,
                "days_until_arrival": sku.days_until_arrival,
            }
            for sku in self._skus
        ]

    def _is_critical(self, sku: SKU) -> bool:
        inventory_position = sku.stock + sku.pending_order
        risk_window = sku.demand_per_day * (sku.supplier_delay_days + 1)
        return inventory_position < risk_window

    def _compute_reward(
        self,
        *,
        units_demanded: int,
        units_unfulfilled: int,
        overstock_count: int,
        critical_count: int,
        rejected_orders: int,
    ) -> float:
        sku_count = len(self._skus)
        if sku_count == 0:
            return STRICT_EPSILON

        service_penalty = units_unfulfilled / units_demanded if units_demanded else 0.0
        overstock_penalty = (overstock_count / sku_count) * 0.20
        critical_penalty = (critical_count / sku_count) * 0.15
        rejection_penalty = (min(rejected_orders, sku_count) / sku_count) * 0.10

        reward = 1.0
        reward -= service_penalty * 0.65
        reward -= overstock_penalty
        reward -= critical_penalty
        reward -= rejection_penalty
        return clamp(reward)
