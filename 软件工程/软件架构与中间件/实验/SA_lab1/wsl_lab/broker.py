
# import eventlet
# eventlet.monkey_patch()

from flask import Flask, request, jsonify
import eventlet
from flask_socketio import SocketIO, join_room
import queue
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", message_queue='redis://127.0.0.1:6379/0', ping_timeout=60)

import logging
import redis
from collections import defaultdict, deque
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
# 存储用户订阅的平台及其消息列表
user_messages = defaultdict(lambda: defaultdict(deque))  # {user_name: {platform_name: deque([message_list])}}
subscribe_map = defaultdict(list)  # {user_name: [platform_list]}
lock = Lock()  # 保护共享资源的锁
# count: int = 0  # 计数器
# 创建线程池
general_executor = ThreadPoolExecutor(max_workers=50)  # 普通优先级线程池
message_queue = queue.Queue()  # 新增消息队列
# 添加 'join' 事件的处理函数
@socketio.on('join')
def handle_join(data):
    user_name = data['user_name']
    join_room(user_name)
    print(f"User {user_name} has joined the room.")

def add_message_to_users(platform_name, message):
    """将消息添加到每个订阅了该平台的用户的消息队列，并通过 SocketIO 发送。"""
    # global count
    with lock:
        for user, platforms in subscribe_map.items():
            if platform_name in platforms:
                user_messages[user][platform_name].append(message)
                socketio.emit(platform_name, message, to=user)  # 发送消息到特定用户
                # print(f"Published message to {user} on {platform_name}, count: {count}")
                # count += 1
        return {"status": "Message published"}, 200

def add_subscription(user_name, platform_name):
    """为用户添加订阅的平台。"""
    with lock:
        if platform_name not in subscribe_map[user_name]:
            subscribe_map[user_name].append(platform_name)
            return {"status": f"User {user_name} subscribed to {platform_name}"}, 200
        else:
            return {"status": "Already subscribed"}, 409

# 定时任务，定期发送消息
def send_messages():
    while True:
        message = message_queue.get()  # 阻塞，直到有消息
        if message is None:  # 允许通过发送 None 来中断
            break
        platform_name, msg = message
        with lock:
            for user, platforms in subscribe_map.items():
                if platform_name in platforms:
                    user_messages[user][platform_name].append(msg)
                    socketio.emit(platform_name, msg, to=user)


# 启动定时任务
eventlet.spawn(send_messages)

@app.route('/publish', methods=['POST'])
def publish():
    data = request.json
    platform_name = data.get('platform')
    message = data.get('message')

    if platform_name and message:
        message_queue.put((platform_name, message))  # 将消息放入队列
        return jsonify({"status": "message queued"}), 200
    return jsonify({"error": "invalid request"}), 400

@app.route('/subscribe', methods=['POST'])
def handle_subscribe():
    """订阅用户到某个平台的主题。"""
    data = request.get_json()
    user_name = data['user']
    platform_name = data['platform']
    print(f"Received subscription request from {user_name} to {platform_name}")
    future = general_executor.submit(add_subscription, user_name, platform_name)
    result, status = future.result()
    return jsonify(result), status

if __name__ == '__main__':
    socketio.run(app=app, host='0.0.0.0', port=9999)
