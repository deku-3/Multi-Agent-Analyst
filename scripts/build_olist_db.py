from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "olist"
DB_PATH = DATA_DIR / "olist.sqlite"

TABLES = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "product_category_name_translation",
}


def load_csvs_to_sqlite(conn: sqlite3.Connection) -> None:
    for filename, table_name in TABLES.items():
        csv_path = DATA_DIR / filename

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")

        print(f"Loading {filename} -> {table_name}")

        df = pd.read_csv(csv_path)

        print(f"  rows: {len(df):,}")
        print(f"  cols: {len(df.columns)}")

        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False,
        )


def create_indexes(conn: sqlite3.Connection) -> None:
    indexes = [
        """
        CREATE INDEX IF NOT EXISTS idx_orders_customer_id
        ON orders(customer_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_orders_purchase_timestamp
        ON orders(order_purchase_timestamp)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_orders_status
        ON orders(order_status)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_order_items_order_id
        ON order_items(order_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_order_items_product_id
        ON order_items(product_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_order_items_seller_id
        ON order_items(seller_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_order_payments_order_id
        ON order_payments(order_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_order_reviews_order_id
        ON order_reviews(order_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_customers_customer_unique_id
        ON customers(customer_unique_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_products_product_id
        ON products(product_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_products_category
        ON products(product_category_name)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_sellers_seller_id
        ON sellers(seller_id)
        """,
    ]

    for sql in indexes:
        conn.execute(sql)

    conn.commit()


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    try:
        load_csvs_to_sqlite(conn)
        create_indexes(conn)

        print(f"\nDatabase created: {DB_PATH}")
        print("Indexes created successfully.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()