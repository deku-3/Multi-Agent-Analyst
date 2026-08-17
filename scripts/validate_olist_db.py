from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "olist" / "olist.sqlite"


def run_query(conn, description, sql):
    result = conn.execute(sql).fetchone()
    print(f"{description}: {result}")


def main():
    conn = sqlite3.connect(DB_PATH)

    try:
        print("\n=== PRIMARY KEY / DUPLICATE CHECKS ===")

        run_query(
            conn,
            "Duplicate order_id",
            """
            SELECT COUNT(*) - COUNT(DISTINCT order_id)
            FROM orders
            """
        )

        run_query(
            conn,
            "Duplicate customer_id",
            """
            SELECT COUNT(*) - COUNT(DISTINCT customer_id)
            FROM customers
            """
        )

        run_query(
            conn,
            "Duplicate customer_unique_id",
            """
            SELECT COUNT(*) - COUNT(DISTINCT customer_unique_id)
            FROM customers
            """
        )

        run_query(
            conn,
            "Duplicate product_id",
            """
            SELECT COUNT(*) - COUNT(DISTINCT product_id)
            FROM products
            """
        )

        run_query(
            conn,
            "Duplicate seller_id",
            """
            SELECT COUNT(*) - COUNT(DISTINCT seller_id)
            FROM sellers
            """
        )

        run_query(
            conn,
            "Duplicate translation category",
            """
            SELECT COUNT(*) - COUNT(DISTINCT product_category_name)
            FROM product_category_name_translation
            """
        )

        print("\n=== NULL CHECKS ===")

        run_query(
            conn,
            "Orders with NULL customer_id",
            """
            SELECT COUNT(*)
            FROM orders
            WHERE customer_id IS NULL
            """
        )

        run_query(
            conn,
            "Items with NULL product_id",
            """
            SELECT COUNT(*)
            FROM order_items
            WHERE product_id IS NULL
            """
        )

        run_query(
            conn,
            "Items with NULL seller_id",
            """
            SELECT COUNT(*)
            FROM order_items
            WHERE seller_id IS NULL
            """
        )

        run_query(
            conn,
            "Products with NULL category",
            """
            SELECT COUNT(*)
            FROM products
            WHERE product_category_name IS NULL
            """
        )

        run_query(
            conn,
            "Orders with NULL purchase timestamp",
            """
            SELECT COUNT(*)
            FROM orders
            WHERE order_purchase_timestamp IS NULL
            """
        )

        print("\n=== ORDER DATE RANGE ===")

        run_query(
            conn,
            "Purchase date range",
            """
            SELECT
                MIN(order_purchase_timestamp),
                MAX(order_purchase_timestamp)
            FROM orders
            """
        )

        print("\n=== ORDER STATUS ===")

        for row in conn.execute(
            """
            SELECT order_status, COUNT(*)
            FROM orders
            GROUP BY order_status
            ORDER BY COUNT(*) DESC
            """
        ):
            print(row)

        print("\n=== CUSTOMER ID RELATIONSHIP ===")

        run_query(
            conn,
            "Customer IDs",
            """
            SELECT COUNT(DISTINCT customer_id)
            FROM customers
            """
        )

        run_query(
            conn,
            "Unique customers",
            """
            SELECT COUNT(DISTINCT customer_unique_id)
            FROM customers
            """
        )

        print("\n=== ITEM PRICE / FREIGHT ===")

        run_query(
            conn,
            "Total item price",
            """
            SELECT SUM(price)
            FROM order_items
            """
        )

        run_query(
            conn,
            "Total freight",
            """
            SELECT SUM(freight_value)
            FROM order_items
            """
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()