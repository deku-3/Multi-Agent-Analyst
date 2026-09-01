from datetime import datetime
from zoneinfo import ZoneInfo


DATASET_MIN_DATE = "2016-09-04"
DATASET_MAX_DATE = "2018-10-17"

PARTIAL_PERIODS = [
    "2016-09",
    "2018-10",
]

METRIC_DEFINITIONS = {
    "sales": "Delivered GMV = SUM(order_items.price)",
    "gmv": "Delivered GMV = SUM(order_items.price)",
    "orders": "COUNT(DISTINCT orders.order_id)",
    "unique_customers": (
        "COUNT(DISTINCT customers.customer_unique_id)"
    ),
    "aov": "Delivered GMV / delivered orders",
    "freight": "SUM(order_items.freight_value)",
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