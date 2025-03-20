from database_utils import DatabaseConnector

db = DatabaseConnector

def save_detected_product(transaction_id, product_name, barcode, count):
    conn = db.ini_db_engine()
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO Detected_Products (transaction_id, product_name, barcode, count)
        VALUES (?, ?, ?, ?)
        """
        cursor.execute(query, (transaction_id, product_name, barcode, count))
        conn.commit()
        conn.close()
        print(f"Saved: {product_name} (x{count}) to database.")
    except Exception as e:
        print("Error inserting data:", e)

