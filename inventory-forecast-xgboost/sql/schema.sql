
CREATE TABLE inventory_forecast(
    id UUID PRIMARY KEY,
    product_id UUID,
    forecast_date DATE,
    predicted_demand NUMERIC,
    safety_stock NUMERIC,
    reorder_point NUMERIC,
    created_at TIMESTAMP
);
