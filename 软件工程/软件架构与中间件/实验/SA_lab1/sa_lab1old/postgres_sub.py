import requests
import threading
import json
from psycopg2 import pool
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()
dbuser = os.getenv('POSTGRES_USER')
dbpassword = os.getenv('POSTGRES_PASSWORD')

# 初始化 PostgreSQL 连接池
connection_pool = pool.SimpleConnectionPool(
    1, 200,  # 最小连接数 1，最大连接数 20
    dbname="postgres",
    user=dbuser,
    password=dbpassword,
    host="localhost",
    port="5432",
)

def handle_message(platform, message):
    """处理单条消息并存储到 PostgreSQL。"""
    conn = None
    cursor = None
    try:
        # 从连接池获取连接
        conn = connection_pool.getconn()
        cursor = conn.cursor()

        # 解析消息并插入数据库
        conversation_id = message.get("conversation_id", str(uuid.uuid4()))
        tokens_used = message.get("tokens_used", 0)
        timestamp = datetime.now()

        cursor.execute(
            "INSERT INTO conversation_logs (platform, conversation_id, tokens_used, timestamp) VALUES (%s, %s, %s, %s)",
            (platform, conversation_id, tokens_used, timestamp),
        )
        conn.commit()
        # print(f"Logged conversation {conversation_id} from platform {platform} to PostgreSQL.")
    except Exception as e:
        print(f"Error handling message in PostgreSQL subscriber: {e}")
    finally:
        # 释放游标和连接
        if cursor:
            cursor.close()
        if conn:
            connection_pool.putconn(conn)

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

def fetch_messages_from_broker(user):
    """从 Broker 拉取消息并处理。"""
    url = f"http://localhost:9999/fetch"
    try:
        response = requests.post(url, json={"user": user})
        if response.status_code == 200:
            data = response.json()
            for platform, messages in data.items():
                print(f"Processing messages from platform: {platform}")
                for message in messages:
                    handle_message(platform, message)
        else:
            print(f"Failed to fetch messages: {response.status_code}")
    except Exception as e:
        print(f"Error fetching messages: {e}")

def scheduled_fetch(user, interval=0.1):
    """定期拉取消息的线程任务。"""
    while True:
        fetch_messages_from_broker(user)
        time.sleep(interval)  # 等待指定时间后再次拉取

if __name__ == "__main__":
    user_name = "postgresql"  # 示例用户
    platform_name = "log"  # 示例平台

    # 用户订阅平台
    subscribe_user_to_platform(user_name, platform_name)

    # 启动一个线程，定期从 Broker 拉取消息
    fetch_thread = threading.Thread(target=scheduled_fetch, args=(user_name,), daemon=True)
    fetch_thread.start()

    # 主线程保持运行，等待用户中断
    try:
        while True:
            time.sleep(1)  # 主线程休眠，保持应用运行
    except KeyboardInterrupt:
        print("PostgreSQL subscriber shutting down.")
        connection_pool.closeall()  # 关闭连接池
