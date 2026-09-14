from datetime import datetime
from zoneinfo import ZoneInfo


DATASET_MIN_DATE = "2016-09-04"
DATASET_MAX_DATE = "2018-10-17"

PARTIAL_PERIODS = [
    "2016-09",
    "2018-10",
]

# NOTE: population filter (order_status = 'delivered') is stated
# explicitly here so the planner's definition cannot drift from the
# SQL Worker's. This is a stopgap until a single shared semantic
# module is consumed by both components.
METRIC_DEFINITIONS = {
    "sales": (
        "Delivered GMV = SUM(order_items.price) "
        "over orders WHERE order_status = 'delivered'"
    ),
    "gmv": (
        "Delivered GMV = SUM(order_items.price) "
        "over orders WHERE order_status = 'delivered'"
    ),
    "orders": (
        "COUNT(DISTINCT orders.order_id) "
        "(delivered orders for sales-related counts)"
    ),
    "unique_customers": (
        "COUNT(DISTINCT customers.customer_unique_id)"
    ),
    "aov": (
        "Delivered GMV / delivered order count "
        "(both over order_status = 'delivered')"
    ),
    "freight": (
        "SUM(order_items.freight_value); "
        "separate from GMV, not part of sales"
    ),
}

SALES_DATE = "orders.order_purchase_timestamp"


def get_current_datetime() -> str:
    return datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).isoformat()


def build_runtime_context() -> dict:
    return {
        "current_datetime": get_current_datetime(),
        "dataset": {
            "name": "Olist Brazilian E-Commerce Dataset",
            "min_date": DATASET_MIN_DATE,
            "max_date": DATASET_MAX_DATE,
            "partial_periods": PARTIAL_PERIODS,
        },
        "metrics": METRIC_DEFINITIONS,
        "sales_date": SALES_DATE,
    }