
import eventlet
from numpy import block
eventlet.monkey_patch()

from flask import Flask, request, jsonify
from threading import Lock
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from flask_socketio import SocketIO, join_room
import logging
import redis

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*", ping_timeout=60)

# 存储用户订阅的平台及其消息列表
user_messages = defaultdict(lambda: defaultdict(deque))  # {user_name: {platform_name: deque([message_list])}}
subscribe_map = defaultdict(list)  # {user_name: [platform_list]}
lock = Lock()  # 保护共享资源的锁
# count: int = 0  # 计数器
# 创建线程池
general_executor = ThreadPoolExecutor(max_workers=50)  # 普通优先级线程池

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

@app.route('/publish', methods=['POST'])
def handle_publish():
    # global count
    """发布消息到某个平台，并将其分发给订阅该平台的用户。"""
    data = request.get_json()
    platform_name = data['platform']
    message = data['message']
    future = general_executor.submit(add_message_to_users, platform_name, message)
    result, status = future.result()
    # print(f"count: {count}")
    # count += 1
    return jsonify(result), status

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
