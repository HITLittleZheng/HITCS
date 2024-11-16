import redis
import json
import time
import psycopg2
from psycopg2 import sql
import threading
from cachetools import LFUCache
from bloom_filter import BloomFilter

# PostgreSQL
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "54322"
}

# Redis缓存配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
CACHE_EXPIRATION = 300  # 缓存过期时间，单位秒

# LFU缓存和布隆过滤器配置
LFU_CACHE_SIZE = 100  # LFU缓存最大容量
BLOOM_FILTER_SIZE = 1000  # 布隆过滤器的大小
ERROR_RATE = 0.01  # 布隆过滤器的错误率

# 初始化Redis连接
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    # 测试连接
    r.ping()
    print("成功连接到Redis服务器。")
except Exception as e:
    print(f"无法连接到Redis服务器。请确保Redis已启动并运行。{e}")
    exit(1)

# 初始化PostgreSQL连接
try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("成功连接到PostgreSQL数据库。")
except Exception as e:
    print(f"无法连接到PostgreSQL数据库。请检查配置。{e}")
    exit(1)

# 初始化LFU缓存和布隆过滤器
lfu_cache = LFUCache(maxsize=LFU_CACHE_SIZE)
bloom_filter = BloomFilter(max_elements=BLOOM_FILTER_SIZE, error_rate=ERROR_RATE)

# 互斥锁
lock = threading.Lock()


def initialize_bloom_filter():
    """
    初始化布隆过滤器，将所有用户ID加载到布隆过滤器中。
    """
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT DISTINCT user_id FROM orders")
            user_ids = cur.fetchall()
            for user_id in user_ids:
                bloom_filter.add(user_id[0])
            print("布隆过滤器已初始化。")
        except Exception as e:
            print(f"初始化布隆过滤器失败：{e}")


# 在程序启动时调用
initialize_bloom_filter()


def get_user_orders(user_id):
    """
    获取用户订单信息。首先尝试从LFU缓存和Redis缓存中获取，如果未命中，则从数据库中获取并缓存。

    :param user_id: 用户ID（字符串）
    :return: 订单列表（列表）
    """
    with lock:
        # 尝试从LFU缓存获取
        if user_id in lfu_cache:
            print(f"LFU缓存命中：获取用户{user_id}的订单数据。")
            return lfu_cache[user_id]

        # 尝试从Redis获取缓存数据
        cache_key = f"user_orders:{user_id}"
        cached_data = r.get(cache_key)
        if cached_data:
            print(f"Redis缓存命中：从Redis获取用户{user_id}的订单数据。")
            orders = json.loads(cached_data, parse_float=str)
            lfu_cache[user_id] = orders  # 存入LFU缓存
            return orders

        # 检查布隆过滤器是否可能存在
        if user_id not in bloom_filter:
            print(f"用户{user_id}不在布隆过滤器中，跳过数据库查询。")
            return []

        # 从PostgreSQL数据库获取数据
        print(f"缓存未命中：从数据库获取用户{user_id}的订单数据。")
        try:
            with conn.cursor() as cur:
                query = sql.SQL(
                    "SELECT order_id, item, price FROM orders WHERE user_id = %s"
                )
                cur.execute(query, (user_id,))
                orders = cur.fetchall()
                orders = [
                    {"order_id": order[0], "item": order[1], "price": order[2]}
                    for order in orders
                ]

                if orders:
                    # 将数据缓存到Redis和LFU缓存
                    r.set(
                        cache_key, json.dumps(orders, default=str), ex=CACHE_EXPIRATION
                    )
                    lfu_cache[user_id] = orders
                    bloom_filter.add(user_id)  # 将用户ID添加到布隆过滤器
                else:
                    print(f"用户{user_id}没有订单数据。")
                return orders
        except Exception as e:
            print(f"从数据库获取数据失败：{e}")
            return []


def add_order(user_id, order):
    """
    向数据库中添加订单，并更新Redis缓存、LFU缓存和布隆过滤器。

    :param user_id: 用户ID（字符串）
    :param order: 订单信息（字典）
    """
    with lock:
        try:
            with conn.cursor() as cur:
                query = sql.SQL(
                    "INSERT INTO orders (user_id, order_id, item, price) VALUES (%s, %s, %s, %s)"
                )
                cur.execute(
                    query, (user_id, order["order_id"], order["item"], order["price"])
                )
                conn.commit()
                print(f"已向用户{user_id}添加新订单。")

                # 更新Redis和LFU缓存
                cache_key = f"user_orders:{user_id}"
                if user_id in lfu_cache:
                    lfu_cache[user_id].append(order)
                else:
                    lfu_cache[user_id] = [order]

                r.set(
                    cache_key,
                    json.dumps(lfu_cache[user_id], default=str),
                    ex=CACHE_EXPIRATION,
                )
                bloom_filter.add(user_id)
                print(f"已更新Redis缓存和LFU缓存。")
        except Exception as e:
            print(f"添加订单失败：{e}")
            conn.rollback()


def delete_order(user_id, order_id):
    """
    从数据库中删除订单，并更新Redis缓存和LFU缓存。

    :param user_id: 用户ID（字符串）
    :param order_id: 订单ID（字符串）
    """
    with lock:
        try:
            with conn.cursor() as cur:
                query = sql.SQL(
                    "DELETE FROM orders WHERE user_id = %s AND order_id = %s"
                )
                cur.execute(query, (user_id, order_id))
                conn.commit()
                print(f"已删除用户{user_id}的订单{order_id}。")

                # 更新Redis和LFU缓存
                cache_key = f"user_orders:{user_id}"
                if user_id in lfu_cache:
                    lfu_cache[user_id] = [
                        order
                        for order in lfu_cache[user_id]
                        if order["order_id"] != order_id
                    ]

                r.set(
                    cache_key,
                    json.dumps(lfu_cache[user_id], default=str),
                    ex=CACHE_EXPIRATION,
                )
                print(f"已更新Redis缓存和LFU缓存。")
        except Exception as e:
            print(f"删除订单失败：{e}")
            conn.rollback()


# 关闭数据库连接
def close_connection():
    conn.close()
    print("已关闭PostgreSQL数据库连接。")


# 测试示例
if __name__ == "__main__":
    user_ids = ["user_1", "user_2", "user_3", "user_4"]

    # 第一次查询，应该从数据库获取并缓存
    print("\n--- 第一次查询订单 ---")
    for uid in user_ids:
        orders = get_user_orders(uid)
        print(f"用户{uid}的订单：{orders}\n")
        time.sleep(0.5)  # 为了输出更清晰

    # 第二次查询，应该从缓存获取
    print("\n--- 第二次查询订单（应命中缓存）---")
    for uid in user_ids:
        orders = get_user_orders(uid)
        print(f"用户{uid}的订单：{orders}\n")
        time.sleep(0.5)

    # 添加新订单
    print("\n--- 添加新订单 ---")
    new_order = {"order_id": "order_1003", "item": "Keyboard", "price": 45}
    add_order("user_1", new_order)

    # 查询更新后的订单
    print("\n--- 查询更新后的订单 ---")
    orders = get_user_orders("user_1")
    print(f"用户user_1的订单：{orders}\n")

    # 删除订单
    print("\n--- 删除订单 ---")
    delete_order("user_2", "order_2001")

    # 查询删除后的订单
    print("\n--- 查询删除后的订单 ---")
    orders = get_user_orders("user_2")
    print(f"用户user_2的订单：{orders}\n")

    # 查询不存在的用户
    print("\n--- 查询不存在的用户 ---")
    orders = get_user_orders("user_5")
    print(f"用户user_5的订单：{orders}\n")

    # 关闭数据库连接
    close_connection()
