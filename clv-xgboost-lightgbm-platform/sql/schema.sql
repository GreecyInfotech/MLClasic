
CREATE TABLE customer_clv (
    customer_id UUID PRIMARY KEY,
    predicted_clv NUMERIC,
    model_name VARCHAR(50),
    created_at TIMESTAMP
);
