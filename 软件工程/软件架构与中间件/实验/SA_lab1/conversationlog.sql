CREATE TABLE conversation_logs (
    id SERIAL PRIMARY KEY,
    platform VARCHAR,
    conversation_id UUID NOT NULL,
    tokens_used INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
