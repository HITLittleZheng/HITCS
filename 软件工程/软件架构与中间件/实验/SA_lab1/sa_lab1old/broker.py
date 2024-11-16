from flask import Flask, request, jsonify
from threading import Lock
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from flask_socketio import SocketIO, emit
import queue
import logging

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)


# 使用字典存储每个用户的订阅平台及其消息列表
user_messages = defaultdict(lambda: defaultdict(deque))  # {user_name: {platform_name: deque([message_list])}}
subscribe_map = defaultdict(list)  # {user_name: [platform_list]}
lock = Lock()  # 保护共享资源的锁

# 创建两个线程池：一个用于 fetch 请求（优先级高），一个用于其他请求
fetch_executor = ThreadPoolExecutor(max_workers=5)  # 高优先级线程池
general_executor = ThreadPoolExecutor(max_workers=20)  # 普通优先级线程池

def add_message_to_users(platform_name, message):
    """将消息添加到每个订阅了该平台的用户的消息队列。"""
    with lock:
        for user, platforms in subscribe_map.items():
            if platform_name in platforms:
                user_messages[user][platform_name].append(message)
                socketio.emit(platform_name, message)
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
    """发布消息到某个平台，并将其分发给订阅该平台的用户。"""
    data = request.get_json()
    platform_name = data['platform']
    message = data['message']

    # 使用普通线程池异步执行发布操作
    future = general_executor.submit(add_message_to_users, platform_name, message)
    result, status = future.result()
    return jsonify(result), status

@app.route('/subscribe', methods=['POST'])
def handle_subscribe():
    """订阅用户到某个平台的主题。"""
    data = request.get_json()
    user_name = data['user']
    platform_name = data['platform']

    # 使用普通线程池异步执行订阅操作
    future = general_executor.submit(add_subscription, user_name, platform_name)
    result, status = future.result()
    return jsonify(result), status

@app.route('/fetch', methods=['POST'])
def fetch_messages():
    """根据用户的订阅，获取尚未处理的消息并返回给用户。"""
    user_name = request.json.get('user')

    # 使用高优先级线程池处理 fetch 请求
    future = fetch_executor.submit(process_fetch, user_name)
    result, status = future.result()
    return jsonify(result), status

def process_fetch(user_name):
    """处理 fetch 请求的具体逻辑。"""
    with lock:
        if user_name not in subscribe_map:
            return {"status": "User not found"}, 404

        user_data = {}
        for platform in subscribe_map[user_name]:
            messages = list(user_messages[user_name][platform])
            user_data[platform] = messages
            user_messages[user_name][platform].clear()  # 清空已拉取的消息

        return user_data, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999, threaded=True)  # 开启多线程支持
