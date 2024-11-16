import socket
import threading
import json
import redis
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Redis 客户端配置
redis_client = redis.Redis(
    host="localhost", 
    port=6379, 
    db=0,
)

# 创建线程池（最大线程数可根据需求调整）
thread_pool = ThreadPoolExecutor(max_workers=10)

def handle_message(conversation):
    """处理接收到的消息并存储到 Redis。"""
    try:
        data = json.loads(conversation)
        conversation_id = data.get("conversation_id", str(uuid.uuid4()))
        messages = data.get("messages", [])

        redis_key = f"conversation:{conversation_id}"
        redis_client.hset(redis_key, mapping={"messages": json.dumps(messages)})
        print(f"Stored conversation {conversation_id} to Redis.")
    except Exception as e:
        print(f"Error handling message in Redis subscriber: {e}")

def submit_message_to_pool(data):
    """提交任务到线程池进行异步处理。"""
    thread_pool.submit(handle_message, data)

def subscribe_to_broker(host="localhost", port=9999):
    """订阅消息中间件，并将消息交由线程池处理。"""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    client.sendall("SUBSCRIBER\n".encode())
    print("Connected to Message Broker as Subscriber.")

    try:
        while True:
            data = client.recv(4096).decode()
            if not data:
                break
            # 将消息提交到线程池处理
            submit_message_to_pool(data)
    except KeyboardInterrupt:
        print("Redis subscriber shutting down.")
    except Exception as e:
        print(f"Error in Redis subscriber: {e}")
    finally:
        client.close()
        thread_pool.shutdown(wait=True)  # 等待所有任务完成

if __name__ == "__main__":
    subscribe_to_broker()
