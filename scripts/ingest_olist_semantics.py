from langchain_core.documents import Document

from src.vectorstore import olist_schema_store


SEMANTIC_DOCUMENT = """
OLIST BUSINESS SEMANTICS AND METRIC DEFINITIONS

DATASET:
Brazilian E-Commerce Public Dataset by Olist.

--------------------------------------------------
CANONICAL SALES / GMV
--------------------------------------------------

Definition:
GMV (sales) = SUM(order_items.price)

Population:
Only orders where orders.order_status = 'delivered'.

Canonical SQL pattern:

SELECT SUM(oi.price)
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
WHERE o.order_status = 'delivered';

Important:
Do NOT define canonical sales/GMV as:

- SUM(order_payments.payment_value)
- SUM(order_items.price + order_items.freight_value)
- SUM(order_items.price) across all order statuses

--------------------------------------------------
ORDER COUNT
--------------------------------------------------

Definition:
COUNT(DISTINCT orders.order_id)

An order may contain multiple order_items, so:

COUNT(*) on order_items != order count.

--------------------------------------------------
UNIQUE CUSTOMER COUNT
--------------------------------------------------

Definition:
COUNT(DISTINCT customers.customer_unique_id)

customer_id identifies the customer record associated with an
order.

customer_unique_id identifies the underlying customer across
orders.

Therefore:

COUNT(DISTINCT customer_id)
is NOT equivalent to
COUNT(DISTINCT customer_unique_id).

--------------------------------------------------
AOV / AVERAGE ORDER VALUE
--------------------------------------------------

Definition:

GMV / delivered order count

Canonical SQL pattern:

SUM(oi.price) * 1.0
/ COUNT(DISTINCT o.order_id)

using delivered orders.

--------------------------------------------------
FREIGHT
--------------------------------------------------

Definition:
SUM(order_items.freight_value)

Freight is separate from canonical GMV.

Do not add freight to GMV unless the user explicitly asks for
order value including freight.

--------------------------------------------------
SALES DATE
--------------------------------------------------

Canonical sales date:
orders.order_purchase_timestamp

Use order_purchase_timestamp for monthly, quarterly, yearly,
and period-based sales analysis unless the user explicitly asks
about another lifecycle date.

Do not use delivery date as the sales date.

--------------------------------------------------
ORDER STATUS
--------------------------------------------------

For canonical sales/GMV analysis:

orders.order_status = 'delivered'

Other statuses include:
shipped
canceled
unavailable
invoiced
processing
created
approved

Do not silently include every status when the question asks for
sales/GMV.

--------------------------------------------------
PRODUCT CATEGORIES
--------------------------------------------------

products.product_category_name contains the original Portuguese
category name.

product_category_name_translation maps the Portuguese category
name to the English category name.

When the user asks for category names in English, join:

products.product_category_name
    ->
product_category_name_translation.product_category_name

and return:

product_category_name_translation.product_category_name_english

--------------------------------------------------
JOIN / FAN-OUT WARNINGS
--------------------------------------------------

orders -> order_items is one-to-many.

orders -> order_payments is one-to-many.

orders -> order_reviews is a separate child relationship.

Do not directly join raw order_items and raw order_payments and
then aggregate item-level sales, because both are child tables
of orders and the join can multiply rows.

Likewise, do not blindly join raw reviews to order_items when
calculating item-level sales.

Aggregate child tables to the required grain before combining
them.

--------------------------------------------------
TIME-PERIOD WARNING
--------------------------------------------------

Observed purchase history in this database:

2016-09-04 through 2018-10-17

September 2016 is a partial month.

October 2018 is a partial month.

Do not treat those as complete months for period-over-period
comparisons unless the user explicitly asks for those partial
periods.

--------------------------------------------------
GRAIN RULES
--------------------------------------------------

orders:
one row per order

order_items:
one row per order item

order_payments:
one row per payment entry

order_reviews:
one row per review record

customers:
one row per order/customer relationship

products:
one row per product

sellers:
one row per seller

geolocation:
multiple rows may exist for a ZIP-code prefix

product_category_name_translation:
one row per translated category

--------------------------------------------------
KNOWN CUSTOMER TRAP
--------------------------------------------------

This database contains:

99,441 distinct customer_id values

96,096 distinct customer_unique_id values

Use customer_unique_id when the question refers to real/unique
customers.

--------------------------------------------------
KNOWN PRODUCT-CATEGORY TRAP
--------------------------------------------------

610 product records have NULL product_category_name.

Do not automatically treat NULL as an actual named category.

--------------------------------------------------
ANALYTICAL PRINCIPLE
--------------------------------------------------

When the user's wording is ambiguous, prefer the canonical
business definitions above rather than inventing a metric.

For "sales", use delivered GMV.

For "orders", count distinct order_id.

For "unique customers", count distinct customer_unique_id.

For "freight", sum freight_value separately.
"""


def main():
    doc = Document(
        page_content=SEMANTIC_DOCUMENT,
        metadata={
            "dataset": "olist",
            "doc_type": "business_semantics",
            "table": "all",
        },
    )

    olist_schema_store.add_documents([doc])

    print("✅ Olist semantic document embedded successfully.")


if __name__ == "__main__":
    main()