@echo off
start python message_broker.py
start python postgres_subscriber.py
start python redis_subscriber.py
timeout /t 1 /nobreak
start streamlit run app.py