from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SKUModel(BaseModel):
    sku_id: str
    stock: int
    demand_per_day: int
    reorder_point: int
    max_stock: int
    supplier_delay_days: int
    pending_order: int = 0
    days_until_arrival: int = 0


class ReorderItem(BaseModel):
    sku_id: str
    quantity: int = Field(gt=0)


class ResetRequest(BaseModel):
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    episode_id: str
    reorder_quantities: List[ReorderItem]


class RewardModel(BaseModel):
    value: float = Field(ge=0.0001, le=0.9999)
    min_value: float = 0.0001
    max_value: float = 0.9999


class SupplyChainObservationModel(BaseModel):
    skus: List[SKUModel]
    day: int
    budget_remaining: float
    stockouts_today: int
    reward: float = Field(ge=0.0001, le=0.9999)
    done: bool
    message: str


class EpisodeStateModel(BaseModel):
    episode_id: str
    difficulty: Literal["easy", "medium", "hard"]
    seed: int
    day: int
    max_days: int
    done: bool
    budget_start: float
    budget_remaining: float
    total_reward: float = Field(ge=0.0001, le=0.9999)
    average_reward: float = Field(ge=0.0001, le=0.9999)
    total_stockouts: int
    stockout_days: int
    overstock_days: int
    critical_days: int
    total_units_demanded: int
    total_units_unfulfilled: int
    service_level: float = Field(ge=0.0001, le=0.9999)
    total_orders_requested: int
    total_orders_placed: int
    total_orders_rejected: int
    rejection_rate: float = Field(ge=0.0001, le=0.9999)
    initial_skus: List[SKUModel]
    skus: List[SKUModel]
    trajectory: List[Dict[str, Any]]


class ResetResponse(BaseModel):
    episode_id: str
    observation: SupplyChainObservationModel
    reward: float = Field(ge=0.0001, le=0.9999)
    done: bool
    state: EpisodeStateModel


class StepResponse(BaseModel):
    episode_id: str
    observation: SupplyChainObservationModel
    reward: float = Field(ge=0.0001, le=0.9999)
    done: bool
    state: EpisodeStateModel


class TaskModel(BaseModel):
    id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    episode_length: int
    success_threshold: float
    default_seed: int


class TaskListResponse(BaseModel):
    tasks: List[TaskModel]


class HealthResponse(BaseModel):
    status: str
