import requests
import threading
import json
from dbutils.pooled_db import PooledDB
from concurrent.futures import ThreadPoolExecutor
import pymysql
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv
import socketio
from flask_socketio import SocketIO

# 加载环境变量
load_dotenv()


# 初始化 PostgreSQL 连接池
connection_pool = PooledDB(
    creator=pymysql,
    db="sharding_db",
    user="root",
    password="root",
    host="127.0.0.1",
    port=3321,
    mincached=2,  # 最小空闲连接数
    maxcached=5,  # 最大空闲连接数
    maxconnections=None,  # 最大连接数
)

user_name = "mysql"  # 替换为实际用户名
# 初始化 SocketIO 客户端
sio = socketio.Client()


# count:int = 0
# 在连接成功后，发送用户名给服务器
@sio.event
def connect():
    print("Connected to message broker.")
    sio.emit("join", {"user_name": user_name})
    subscribe_user_to_platform(user_name, "log")  # 订阅特定平台


@sio.event
def disconnect():
    print("Disconnected from message broker.")


@sio.on("log")  # 监听具体的消息平台
def handle_message(message):
    """处理单条消息并存储到 PostgreSQL。"""
    # global count
    conn = None
    cursor = None
    try:
        # print(f"count: {count}")
        # count += 1
        conn = connection_pool.connection()
        cursor = conn.cursor()

        # print(f"Received message: {message}")
        conversation_id = message.get("conversation_id", str(uuid.uuid4()))
        tokens_used = message.get("tokens_used", 0)
        chat_history = message.get("chat_history", "")
        chat_history = json.dumps(chat_history)
        title = message.get("title", "")

        cursor.execute(
            """
            INSERT INTO t_chat_history (conversation_id, title, chat_history, token_usage)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                chat_history = VALUES(chat_history),
                token_usage = VALUES(token_usage)
            """,
            (conversation_id, title, chat_history, tokens_used),
        )
        conn.commit()

        # print(f"Logged conversation {conversation_id} from platform 'log' to PostgreSQL.")
    except Exception as e:
        print(f"Error handling message in MySQL subscriber: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def subscribe_user_to_platform(user, platform):
    """订阅用户到指定平台。"""
    url = "http://localhost:9999/subscribe"
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
    sio.connect("http://localhost:9999", wait_timeout=10)

    sio.wait()


if __name__ == "__main__":
    listener_thread = threading.Thread(target=start_listener, args=(user_name,))
    listener_thread.start()
