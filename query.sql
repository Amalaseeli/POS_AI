CREATE TABLE Detected_Products (
    id INT IDENTITY(1,1) PRIMARY KEY,
    transaction_id INT,
    product_name VARCHAR(255),
    barcode VARCHAR(50),
    count INT,
    timestamp DATETIME DEFAULT GETDATE()
);