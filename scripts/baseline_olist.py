from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "olist" / "olist.sqlite"


QUERIES = {
    "Delivered GMV": """
        SELECT SUM(oi.price)
        FROM orders o
        JOIN order_items oi
            ON o.order_id = oi.order_id
        WHERE o.order_status = 'delivered'
    """,

    "Delivered Orders": """
        SELECT COUNT(DISTINCT order_id)
        FROM orders
        WHERE order_status = 'delivered'
    """,

    "Unique Customers": """
        SELECT COUNT(DISTINCT c.customer_unique_id)
        FROM orders o
        JOIN customers c
            ON o.customer_id = c.customer_id
        WHERE o.order_status = 'delivered'
    """,

    "AOV": """
        SELECT
            SUM(oi.price) * 1.0
            / COUNT(DISTINCT o.order_id)
        FROM orders o
        JOIN order_items oi
            ON o.order_id = oi.order_id
        WHERE o.order_status = 'delivered'
    """,

    "Delivered Freight": """
        SELECT SUM(oi.freight_value)
        FROM orders o
        JOIN order_items oi
            ON o.order_id = oi.order_id
        WHERE o.order_status = 'delivered'
    """,
}


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    try:
        for name, sql in QUERIES.items():
            result = conn.execute(sql).fetchone()[0]
            print(f"{name}: {result}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()