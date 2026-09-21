from datetime import datetime
from zoneinfo import ZoneInfo


DATASET_MIN_DATE = "2016-09-04"
DATASET_MAX_DATE = "2018-10-17"

# The dataset's timestamp range runs to 2018-10-17, but order collection
# effectively STOPS at the end of August 2018. Verified order counts:
#   2018-08: 6,512   2018-09: 16   2018-10: 4
# So Sep/Oct 2018 are not "partial" months - they are essentially empty
# trailing noise. The last genuinely complete month is 2018-08.
#
# Relative periods and trend/decline analysis should treat 2018-08 as the
# effective end of usable data, NOT 2018-10-17.
DATASET_EFFECTIVE_MAX_DATE = "2018-08-31"
LAST_COMPLETE_PERIOD = "2018-08"

PARTIAL_PERIODS = [
    "2016-09",   # sparse start
    "2018-09",   # near-empty tail (16 orders)
    "2018-10",   # near-empty tail (4 orders)
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

# Ground-truth quirks verified by a full DB audit. These are authoritative
# facts the planner (and eventually the SQL Worker) must respect so it does
# not silently produce inconsistent or incomplete numbers.
DATA_FACTS = [
    # Terminology synonym: "customer rating(s)", "rating(s)", "review
    # score(s)" all refer to the SAME column - avoid burning a query
    # discovering this by trial and error.
    "'Customer rating(s)', 'rating(s)', and 'review score(s)' all refer "
    "to the SAME column: order_reviews.review_score. There is no "
    "separate 'rating' column - do not query for one.",

    # Orphan orders: 775 orders have NO rows in order_items (mostly
    # unavailable/canceled). This makes the order denominator ambiguous.
    "775 orders have no order_items. Sales/GMV/AOV counts use orders "
    "that HAVE items (inner join order_items). Operational order counts "
    "('how many orders were placed') use the orders table directly. "
    "State which basis is used.",

    # Category nulls: 610 products have NULL product_category_name.
    "610 products have a NULL category. Category-level analysis must "
    "either bucket these as 'uncategorized' or explicitly note they are "
    "excluded - never silently drop them.",

    # Category names are Portuguese; join translation for English labels.
    "product.product_category_name is Portuguese. Join "
    "product_category_name_translation for English labels (2 categories "
    "have no translation).",

    # Review-score skew justifies quartile-based 'low'/'high'.
    "review_score is heavily left-skewed (5 is most common; 3 is rarer "
    "than 4). 'Low'/'high' review thresholds must be quartile-based, "
    "never a fixed midpoint like < 3.",

    # Geography concentration.
    "Orders are dominated by Sao Paulo (SP ~42% of all orders). "
    "'Which region' analysis is effectively SP vs the rest.",

    # Repeat rate is genuinely tiny.
    "Repeat-purchase rate is ~3% (2,997 of 96,096 unique customers have "
    ">1 order). Report the RATE, not just a raw count.",

    # sales != payment.
    "SUM(order_items.price) (GMV, ~R$13.6M) != SUM(payment_value) "
    "(~R$16.0M). payment_value includes freight and installments and is "
    "NOT sales.",
]


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
            "effective_max_date": DATASET_EFFECTIVE_MAX_DATE,
            "last_complete_period": LAST_COMPLETE_PERIOD,
            "partial_periods": PARTIAL_PERIODS,
        },
        "metrics": METRIC_DEFINITIONS,
        "sales_date": SALES_DATE,
        "data_facts": DATA_FACTS,
    }