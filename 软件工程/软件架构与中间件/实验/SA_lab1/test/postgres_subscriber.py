import socket
import threading
import json
import psycopg2
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
dbuser = os.getenv('POSTGRES_USER')
dbpassword = os.getenv('POSTGRES_PASSWORD')

# PostgreSQL 连接配置
pg_conn = psycopg2.connect(
    dbname="postgres",
    user=dbuser,
    password=dbpassword,
    host="localhost",
    port="5432",
)

pg_cursor = pg_conn.cursor()


def handle_message(message):
    try:
        data = json.loads(message)
        conversation_id = data.get("conversation_id", str(uuid.uuid4()))
        tokens_used = data.get("tokens_used", 0)
        timestamp = datetime.now()

        pg_cursor.execute(
            "INSERT INTO conversation_logs (conversation_id, tokens_used, timestamp) VALUES (%s, %s, %s)",
            (conversation_id, tokens_used, timestamp),
        )
        pg_conn.commit()
        print(f"Logged conversation {conversation_id} to PostgreSQL.")
    except Exception as e:
        print(f"Error handling message in PostgreSQL subscriber: {e}")


def subscribe_to_broker(host="localhost", port=9999):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    client.sendall("SUBSCRIBER\n".encode())
    print("Connected to Message Broker as Subscriber.")

    try:
        while True:
            data = client.recv(4096).decode()
            if not data:
                break
            handle_message(data)
    except KeyboardInterrupt:
        print("PostgreSQL subscriber shutting down.")
    except Exception as e:
        print(f"Error in PostgreSQL subscriber: {e}")
    finally:
        client.close()
        pg_cursor.close()
        pg_conn.close()


if __name__ == "__main__":
    subscribe_to_broker()
