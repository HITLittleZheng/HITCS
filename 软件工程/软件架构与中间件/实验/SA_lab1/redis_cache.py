import redis
from redisbloom.client import Client as BloomClient
from threading import Lock
from dbutils.pooled_db import PooledDB
import pymysql
import json
import time

# 初始化Redis和布隆过滤器
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
# bloom_client = BloomClient(host="localhost", port=6379)
# filter_name = "conversation_filter"
# error_rate = 0.01  # 误判率 1%
# capacity = 1000  # 预期存储容量

# # 创建布隆过滤器
# bloom_client.bfCreate(filter_name, error_rate, capacity)

# 互斥锁字典，用于每个 conversation_id
mutex_locks = {}
dbuser = "root"
dbpassword = "root"

# 初始化 MySQL 连接池
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


# 插入新对话记录，刷新第一页缓存
def update_cache():
    conn = connection_pool.connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT conversation_id, title, chat_history, timestamp, token_usage
        FROM t_chat_history
        ORDER BY timestamp DESC
        LIMIT 10
    """
    )

    recent_conversations = cursor.fetchall()
    print(recent_conversations)
    if not recent_conversations:

        return []
    # 更新第一页缓存
    page_data = {}
    for conv in recent_conversations:
        conversation_id, title, chat_history, timestamp, token_usage = conv
        chat_history_json = json.dumps(json.loads(chat_history))

        # 缓存第一页数据
        page_data[conversation_id] = json.dumps(
            {
                "chat_history": chat_history_json,
                "title": title,
                "timestamp": timestamp.isoformat(),
            }
        )

        # 更新每个 conversation 的单独缓存
        redis_client.hset(
            f"conversation:{conversation_id}",
            mapping={
                "chat_history": chat_history_json,
                "title": title,
                "timestamp": timestamp.isoformat(),
            },
        )
        redis_client.expire(f"conversation:{conversation_id}", 3600)

    # 更新第一页缓存并设置过期时间
    redis_client.hset(f"conversations:page:1", mapping=page_data)
    redis_client.expire(f"conversations:page:1", 3600)

    conn.commit()

    return recent_conversations


# 分页查询全部记录
def get_all_conversations(page=1, page_size=10):
    print(page)
    conn = connection_pool.connection()

    cache_key = f"conversations:page:{page}"
    # cached_data = redis_client.hgetall(cache_key)

    # 缓存里有的话就直接返回
    # if cached_data and not cached_data.get(b"empty"):
    #     # 解码缓存数据
    #     for key in cached_data:
    #         cached_data[key] = json.loads(cached_data[key].decode("utf-8"))

    #     return cached_data

    # 数据库查询分页数据
    cursor = conn.cursor()
    offset = (page - 1) * page_size
    cursor.execute(
        "SELECT conversation_id, title, chat_history, timestamp, token_usage FROM t_chat_history ORDER BY timestamp DESC LIMIT %s OFFSET %s",
        (page_size, offset),
    )
    conversations = cursor.fetchall()

    if not conversations:
        # 如果该页没有数据，缓存空标记
        redis_client.hset(cache_key, mapping={"empty": "true"})
        redis_client.expire(cache_key, 3600)
        return []
    # 将查询的数据写入缓存
    page_data = {}
    for conv in conversations:
        conversation_id, title, chat_history, timestamp, token_usage = conv
        chat_history_json = json.dumps(json.loads(chat_history))

        page_data[conversation_id] = json.dumps(
            {
                "chat_history": chat_history_json,
                "title": title,
                "timestamp": timestamp.isoformat(),
            }
        )

        # 更新每个 conversation 的单独缓存
        redis_client.hset(
            f"conversation:{conversation_id}",
            mapping={
                "chat_history": chat_history_json,
                "title": title,
                "timestamp": timestamp.isoformat(),
            },
        )
        redis_client.expire(f"conversation:{conversation_id}", 3600)

    # 缓存当前页并设置过期时间
    redis_client.hset(cache_key, mapping=page_data)
    redis_client.expire(cache_key, 3600)

    return conversations


# 3. 通过conversation_id查询记录,添加互斥锁机制
def get_conversation_by_id(conversation_id):
    # 从 Redis 中获取哈希表数据
    cached_data = redis_client.hgetall(f"conversation:{conversation_id}")

    # 如果缓存中存在数据，将其转换为字典形式并返回
    if cached_data:
        return {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in cached_data.items()
        }

    # 获取或创建该conversation_id的互斥锁
    lock = mutex_locks.setdefault(conversation_id, Lock())

    with lock:
        # 再次检查缓存，防止其他线程已更新
        cached_data = redis_client.hgetall(f"conversation:{conversation_id}")
        if cached_data:
            return {
                key.decode("utf-8"): value.decode("utf-8")
                for key, value in cached_data.items()
            }

        # 从数据库中查询
        conn = connection_pool.connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT conversation_id, title, chat_history, timestamp, token_usage FROM t_chat_history WHERE conversation_id = %s",
            (conversation_id,),
        )
        conversation = cursor.fetchone()
        cursor.close()
        conn.close()

        if conversation:
            conversation_id, title, chat_history, timestamp, token_usage = conversation
            chat_history_json = json.dumps(json.loads(chat_history))
            conversation_data = {
                "chat_history": chat_history_json,
                "title": title,
                "timestamp": timestamp.isoformat(),
                "token_usage": token_usage,
            }

            # 将数据写入缓存
            redis_client.hset(
                f"conversation:{conversation_id}", mapping=conversation_data
            )
            redis_client.expire(f"conversation:{conversation_id}", 3600)

            return conversation_data
        else:
            # 如果数据库中没有数据，设置空标记
            redis_client.hset(
                f"conversation:{conversation_id}", mapping={"empty": "true"}
            )
            redis_client.expire(f"conversation:{conversation_id}", 3600)
            return None


def test_set_empty_flag_in_redis_when_data_not_exist():
    """
    测试查询一个不存在的会话 ID，是否在 Redis 中设置了空标记
    """
    conversation_id = "11"

    # 调用函数
    result = get_conversation_by_id(conversation_id)

    # 检查返回值为 None
    if result is not None:
        print("返回值不为None")
    else:
        print("返回值为None")

    # 检查 Redis 中是否设置了空标记
    cached_data = redis_client.hgetall(f"conversation:{conversation_id}")
    if cached_data[b"empty"] == b"true":
        print("空标记设置成功")
    else:
        print("空标记设置失败")
import threading

def test_mutex_lock_prevents_multiple_db_queries():
    """
    测试多个线程同时查询一个不存在的会话 ID，是否只有一个请求访问数据库
    """
    conversation_id = "111"


    # 记录数据库查询次数
    conn = connection_pool.connection()
    cursor = conn.cursor()
    execute_count = {"count": 0}

    # 定义线程目标函数
    def target():
        result = get_conversation_by_id(conversation_id)
        print(result)
        if result is not None:
            print("返回值不为None,查询了redis")

    # 启动多个线程同时调用函数
    thread_count = 5
    threads = [threading.Thread(target=target) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    # 测试
    test_mutex_lock_prevents_multiple_db_queries()
