import socket
import threading
import json
from psycopg2 import pool
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

count:int = 1

# 加载环境变量
load_dotenv()
dbuser = os.getenv('POSTGRES_USER')
dbpassword = os.getenv('POSTGRES_PASSWORD')

# 初始化 PostgreSQL 连接池
connection_pool = pool.SimpleConnectionPool(
    1, 2000,  # 最小连接数 1，最大连接数 20
    dbname="postgres",
    user=dbuser,
    password=dbpassword,
    host="localhost",
    port="5432",
)

def handle_message(message):
    conn = None
    cursor = None
    global count
    try:
        # 从连接池获取连接
        conn = connection_pool.getconn()
        cursor = conn.cursor()

        # 处理消息并插入到数据库
        data = json.loads(message)
        conversation_id = data.get("conversation_id", str(uuid.uuid4()))
        tokens_used = data.get("tokens_used", 0)
        timestamp = datetime.now()

        cursor.execute(
            "INSERT INTO conversation_logs (conversation_id, tokens_used, timestamp) VALUES (%s, %s, %s)",
            (conversation_id, tokens_used, timestamp),
        )
        conn.commit()
        # print(f"{count}Logged conversation {conversation_id} to PostgreSQL.")
        count = count + 1
    except Exception as e:
        print(f"Error handling message in PostgreSQL subscriber: {e}")
    finally:
        # 确保游标和连接被正确关闭和释放回池中
        if cursor:
            cursor.close()
        if conn:
            connection_pool.putconn(conn)

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
        connection_pool.closeall()  # 关闭连接池

if __name__ == "__main__":
    subscribe_to_broker()
