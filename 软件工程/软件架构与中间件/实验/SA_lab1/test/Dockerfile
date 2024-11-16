# 使用Python 3.12基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . .

# 安装所需的Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装Redis和PostgreSQL客户端工具
RUN apt-get update && apt-get install -y redis postgresql && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 暴露默认端口：Redis(6379)，PostgreSQL(5432)，Streamlit(8501)
EXPOSE 6379 5432 8501

# 设置环境变量，PostgreSQL账号和密码
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# 初始化Redis和PostgreSQL数据库
CMD service redis-server start && \
    service postgresql start && \
    su postgres -c "psql -c \"ALTER USER postgres PASSWORD 'postgres';\"" && \
    su postgres -c "psql -c \"CREATE DATABASE postgres;\"" && \
    waitress-serve --port=9999 --threads=50 --connection-limit=5000 --backlog=800 broker:app & \
    python redis_subscriber.py & \
    python postgres_sub.py & \
    streamlit run app.py --server.port 8501 --server.address=0.0.0.0
