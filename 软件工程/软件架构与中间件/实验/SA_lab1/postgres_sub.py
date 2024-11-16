import requests
import threading
import json
from psycopg2 import pool
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv
import socketio

# 加载环境变量
load_dotenv()
dbuser = os.getenv('POSTGRES_USER')
dbpassword = os.getenv('POSTGRES_PASSWORD')

# 初始化 PostgreSQL 连接池
connection_pool = pool.SimpleConnectionPool(
    20, 1000,  # 最小连接数 1，最大连接数 20
    dbname="postgres",
    user=dbuser,
    password=dbpassword,
    host="127.0.0.1",
    port="5432",
)

user_name = "postgres"  # 替换为实际用户名
# 初始化 SocketIO 客户端
sio = socketio.Client()
count:int = 0
# 在连接成功后，发送用户名给服务器
@sio.event
def connect():
    print("Connected to message broker.")
    sio.emit('join', {'user_name': user_name})
    subscribe_user_to_platform(user_name, "log")  # 订阅特定平台

@sio.event
def disconnect():
    print("Disconnected from message broker.")

@sio.on('log')  # 监听具体的消息平台
def handle_message(message):
    """处理单条消息并存储到 PostgreSQL。"""
    global count
    conn = None
    cursor = None
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()

        # print(f"Received message: {message}")
        conversation_id = message.get("conversation_id", str(uuid.uuid4()))
        tokens_used = message.get("tokens_used", 0)
        timestamp = datetime.now()

        cursor.execute(
            "INSERT INTO conversation_logs (platform, conversation_id, tokens_used, timestamp) VALUES (%s, %s, %s, %s)",
            ("log", conversation_id, tokens_used, timestamp),
        )
        conn.commit()
        print(f"count: {count}")
        count += 1
        # print(f"Logged conversation {conversation_id} from platform 'log' to PostgreSQL.")
    except Exception as e:
        print(f"Error handling message in PostgreSQL subscriber: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            connection_pool.putconn(conn)

def subscribe_user_to_platform(user, platform):
    """订阅用户到指定平台。"""
    url = "http://172.21.198.117:9999/subscribe"
    payload = {"user": user, "platform": platform}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"User {user} subscribed to {platform}.")
        else:
            print(f"Subscription failed: {response.json()}")
    except Exception as e:
        print(f"Error subscribing user: {e}")

def start_listener(user):
    """启动 SocketIO 客户端监听消息。"""
    sio.connect('http://172.21.198.117:9999', wait_timeout=10)
    
    sio.wait()

if __name__ == "__main__":
    
    listener_thread = threading.Thread(target=start_listener, args=(user_name,))
    listener_thread.start()
