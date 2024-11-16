import requests
import threading
import json
import redis
import uuid
from concurrent.futures import ThreadPoolExecutor
import time

# Redis 客户端配置
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# 创建线程池（最大线程数可根据需求调整）
thread_pool = ThreadPoolExecutor(max_workers=10)

def handle_message(platform, message): 
    """处理接收到的消息并存储到 Redis。"""
    try:
        conversation_id = message.get("conversation_id", str(uuid.uuid4()))
        messages = message.get("messages", [])
        title = message.get("title", "")
        redis_key = f"conversation:{conversation_id}"
        timestamp = int(time.time())
        redis_client.hset(redis_key, mapping={"messages": json.dumps(messages), "title": title, "timestamp": timestamp})
        redis_client.zadd("conversations_by_timestamp", {redis_key: timestamp})
        print(f"Stored conversation {conversation_id} under platform {platform} in Redis.")
    except Exception as e:
        print(f"Error handling message in Redis subscriber: {e}")
        
def fetch_messages_from_broker(user):
    """从 Broker 拉取消息并提交到线程池处理。"""
    url = f"http://localhost:9999/fetch"
    try:
        response = requests.post(url, json={"user": user})
        if response.status_code == 200:
            data = response.json()
            for platform, messages in data.items():
                print(f"Processing messages from platform: {platform}")
                for message in messages:
                    thread_pool.submit(handle_message, platform, message)
        else:
            print(f"Failed to fetch messages: {response.status_code}")
    except Exception as e:
        print(f"Error fetching messages: {e}")

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

def scheduled_fetch(user, interval=0.1):
    """定期从 Broker 拉取消息的线程任务。"""
    while True:
        fetch_messages_from_broker(user)
        time.sleep(interval)

if __name__ == "__main__":
    user_name = "redis"  # 示例用户
    platform_name = "log"  # 示例平台

    # 用户订阅平台
    subscribe_user_to_platform(user_name, platform_name)

    # 启动一个线程，定期从 Broker 拉取消息
    fetch_thread = threading.Thread(target=scheduled_fetch, args=(user_name,), daemon=True)
    fetch_thread.start()

    # 主线程保持运行，等待用户中断
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Redis subscriber shutting down.")
        thread_pool.shutdown(wait=True)  # 等待所有任务完成
