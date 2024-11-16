import random
import time
from datetime import datetime

# 定义用户ID和操作类型的范围
USER_COUNT = 10000  # 1万用户
RECORD_COUNT = 1000000  # 100万条操作记录

# 定义操作类型（可以根据需要修改）
OPERATIONS = ['login', 'logout', 'view', 'click', 'purchase', 'error']

# 日志文件名
LOG_FILE = "user_operations.log"

# 生成一个随机的日志记录
def generate_log_record(user_id):
    operation = random.choice(OPERATIONS)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 模拟日志格式：时间戳 用户ID 操作类型
    return f"{timestamp} | user_{user_id} | {operation}\n"

# 将记录写入日志文件
def generate_log_file():
    with open(LOG_FILE, 'w') as f:
        for _ in range(RECORD_COUNT):
            user_id = random.randint(1, USER_COUNT)
            log_record = generate_log_record(user_id)
            f.write(log_record)

if __name__ == "__main__":
    start_time = time.time()
    print("开始生成日志记录...")
    generate_log_file()
    end_time = time.time()
    print(f"日志生成完成，总耗时：{end_time - start_time:.2f}秒")
