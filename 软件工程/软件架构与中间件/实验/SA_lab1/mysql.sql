CREATE TABLE t_chat_history (
    id BIGINT,               
    conversation_id VARCHAR(100) NOT NULL,   
    title VARCHAR(100),       
    chat_history JSON, 
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    token_usage INT,
    UNIQUE(conversation_id)
);
