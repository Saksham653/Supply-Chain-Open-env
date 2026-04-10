import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SKU:
    sku_id: str
    stock: int
    demand_per_day: int
    reorder_point: int
    max_stock: int
    supplier_delay_days: int
    pending_order: int = 0
    days_until_arrival: int = 0


class DemandSimulator:
    """Simulates daily demand for each SKU with a seeded RNG."""

    def __init__(self, difficulty: str, rng: random.Random):
        assert difficulty in ("easy", "medium", "hard")
        self.difficulty = difficulty
        self.rng = rng

    def get_demand(self, sku: SKU) -> int:
        base = sku.demand_per_day

        if self.difficulty == "easy":
            return base

        if self.difficulty == "medium":
            variation = self.rng.uniform(0.8, 1.2)
            return max(1, int(round(base * variation)))

        if self.rng.random() < 0.2:
            spike = self.rng.uniform(1.5, 2.5)
            return max(1, int(round(base * spike)))

        variation = self.rng.uniform(0.8, 1.2)
        return max(1, int(round(base * variation)))


class SupplierSimulator:
    """Simulates delivery delays with a seeded RNG."""

    def __init__(self, difficulty: str, rng: random.Random):
        assert difficulty in ("easy", "medium", "hard")
        self.difficulty = difficulty
        self.rng = rng

    def get_delivery_delay(self, sku: SKU) -> int:
        base = sku.supplier_delay_days

        if self.difficulty == "easy":
            return base

        if self.difficulty == "medium":
            return max(1, base + self.rng.randint(-1, 1))

        extra = self.rng.randint(1, 3) if self.rng.random() < 0.2 else 0
        return max(1, base + extra)


class SKUFactory:
    """Generates seeded SKU portfolios by difficulty."""

    DIFFICULTY_CONFIG = {
        "easy": {
            "num_skus": 1,
            "delay_range": (1, 1),
            "demand_range": (6, 10),
            "stock_range": (40, 55),
        },
        "medium": {
            "num_skus": 10,
            "delay_range": (1, 3),
            "demand_range": (4, 12),
            "stock_range": (35, 55),
        },
        "hard": {
            "num_skus": 10,
            "delay_range": (2, 5),
            "demand_range": (6, 14),
            "stock_range": (40, 65),
        },
    }

    @classmethod
    def generate(cls, difficulty: str, rng: random.Random) -> List[SKU]:
        cfg = cls.DIFFICULTY_CONFIG[difficulty]
        skus: List[SKU] = []

        for index in range(cfg["num_skus"]):
            delay = rng.randint(*cfg["delay_range"])
            demand = rng.randint(*cfg["demand_range"])
            stock = rng.randint(*cfg["stock_range"])
            skus.append(
                SKU(
                    sku_id=f"SKU_{chr(65 + index)}",
                    stock=stock,
                    demand_per_day=demand,
                    reorder_point=demand * (delay + 2),
                    max_stock=200,
                    supplier_delay_days=delay,
                )
            )

        return skus


class ReorderProcessor:
    """
    Applies reorder decisions and processes in-flight deliveries.
    Cost per unit = $10 flat. Only one pending order per SKU at a time.
    """

    UNIT_COST = 10.0

    def __init__(self, supplier: SupplierSimulator):
        self.supplier = supplier

    def process_arrivals(self, skus: List[SKU]) -> List[Dict[str, int | str]]:
        arrivals: List[Dict[str, int | str]] = []

        for sku in skus:
            if sku.days_until_arrival <= 0:
                continue

            sku.days_until_arrival -= 1
            if sku.days_until_arrival == 0:
                received_units = min(sku.pending_order, sku.max_stock - sku.stock)
                sku.stock += received_units
                arrivals.append(
                    {
                        "sku_id": sku.sku_id,
                        "status": "received",
                        "requested_quantity": sku.pending_order,
                        "received_quantity": received_units,
                    }
                )
                sku.pending_order = 0

        return arrivals

    def place_orders(
        self,
        skus: List[SKU],
        reorder_map: Dict[str, int],
        budget: float,
    ) -> Tuple[float, List[Dict[str, int | float | str]]]:
        receipts: List[Dict[str, int | float | str]] = []

        for sku in skus:
            qty = int(reorder_map.get(sku.sku_id, 0))
            if qty <= 0:
                continue

            if sku.pending_order > 0:
                receipts.append(
                    {
                        "sku_id": sku.sku_id,
                        "status": "rejected",
                        "reason": "pending order already in flight",
                        "quantity": qty,
                        "cost": qty * self.UNIT_COST,
                    }
                )
                continue

            qty = min(qty, max(0, sku.max_stock - (sku.stock + sku.pending_order)))
            if qty <= 0:
                receipts.append(
                    {
                        "sku_id": sku.sku_id,
                        "status": "rejected",
                        "reason": "warehouse capacity reached",
                        "quantity": 0,
                        "cost": 0.0,
                    }
                )
                continue

            cost = qty * self.UNIT_COST
            if cost > budget:
                receipts.append(
                    {
                        "sku_id": sku.sku_id,
                        "status": "rejected",
                        "reason": "insufficient budget",
                        "quantity": qty,
                        "cost": cost,
                    }
                )
                continue

            budget -= cost
            sku.pending_order = qty
            sku.days_until_arrival = self.supplier.get_delivery_delay(sku)
            receipts.append(
                {
                    "sku_id": sku.sku_id,
                    "status": "placed",
                    "quantity": qty,
                    "cost": cost,
                    "arrives_in_days": sku.days_until_arrival,
                }
            )

        return budget, receipts


class DailyStepRunner:
    """Runs daily demand and returns detailed demand metrics."""

    def __init__(self, demand_sim: DemandSimulator):
        self.demand_sim = demand_sim

    def run(self, skus: List[SKU]) -> Dict[str, object]:
        stockouts = 0
        units_demanded = 0
        units_unfulfilled = 0
        demand_details: List[Dict[str, int | bool | str]] = []

        for sku in skus:
            demand = self.demand_sim.get_demand(sku)
            fulfilled = min(sku.stock, demand)
            lost = max(0, demand - sku.stock)
            sku.stock = max(0, sku.stock - demand)

            units_demanded += demand
            units_unfulfilled += lost
            if lost > 0:
                stockouts += 1

            demand_details.append(
                {
                    "sku_id": sku.sku_id,
                    "demand": demand,
                    "fulfilled": fulfilled,
                    "lost_sales": lost,
                    "stockout": lost > 0,
                    "ending_stock": sku.stock,
                }
            )

        return {
            "stockouts": stockouts,
            "units_demanded": units_demanded,
            "units_unfulfilled": units_unfulfilled,
            "demand_details": demand_details,
        }
