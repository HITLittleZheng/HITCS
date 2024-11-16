import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import uuid
import json
import redis
import os
import time
import requests  # 使用 requests 代替 socket
from dotenv import load_dotenv
from chain import build_app, generate, generate_title, num_tokens_from_string
import redis_cache
from collections import OrderedDict

# Redis 客户端配置（用于即时更新历史）
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# def get_history():
#     """获取 Redis 中存储的对话历史，并按 timestamp 降序排序。"""
#     try:
#         keys = redis_client.zrevrange("conversations_by_timestamp", 0, -1)
#         history = []
#         for key in keys:
#             convo = redis_client.hgetall(key)
#             history.append(
#                 {
#                     "conversation_id": key.decode().split(":")[1],
#                     "messages": json.loads(convo.get(b"messages", b"[]").decode()),
#                     "title": convo.get(b"title", b"").decode(),
#                     "timestamp": int(convo.get(b"timestamp", b"0").decode())
#                 }
#             )
#         return history
#     except Exception as e:
#         st.sidebar.error(f"Error fetching history: {e}")
#         return []

def get_history(page):
    history = []
    if page == 1:
        conversations = redis_cache.update_cache()
    else:
        print("Fetching from cache")
        conversations = redis_cache.get_all_conversations(page)
    try:
        for conv in conversations:
            conversation_id, title, chat_history, timestamp, token_usage = conv
            chat_history = json.loads(chat_history)
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            history.append(
                {
                "conversation_id": conversation_id,
                "title": title,
                "chat_history": chat_history,
                "timestamp": timestamp,
                "token_usage": token_usage
                }
            )
        return history
    except Exception as e:
        st.sidebar.error(f"Error fetching history: {e}")
        return []

def publish_message(platform, message):
    """通过 HTTP 请求将消息发布到 Broker。"""
    url = "http://localhost:9999/publish"
    payload = {
        "platform": platform,
        "message": message
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Published message to {platform}")
        else:
            st.error(f"Failed to publish message: {response.json()}")
    except Exception as e:
        st.error(f"Error publishing message: {e}")

# 初始化 Streamlit 界面
st.title("软件架构小助手")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "logs" not in st.session_state:
    st.session_state.logs = []
if "title" not in st.session_state:
    st.session_state.title = ""
if "page" not in st.session_state:
    st.session_state.page = 1
# 初始化或更新有序集合的 history_list
if "history_list" not in st.session_state:
    st.session_state.history_list = OrderedDict()

# 获取历史记录
history = get_history(1)

# 将获取的历史记录加入到有序集合中
for convo in history:
    convo_id = convo["conversation_id"]
    if convo_id not in st.session_state.history_list:
        st.session_state.history_list[convo_id] = convo

# 按时间戳排序有序集合
st.session_state.history_list = OrderedDict(
    sorted(
        st.session_state.history_list.items(),
        key=lambda x: x[1]["timestamp"],
        reverse=True,
    )
)

# 侧边栏显示历史对话
st.sidebar.header("历史对话")

if st.sidebar.button("✨新建对话", use_container_width=True):
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.title = ""


for convo_id, convo in st.session_state.history_list.items():
    if st.sidebar.button(f"{convo['title']}", key=convo_id, use_container_width=True):
        st.session_state.conversation_id = convo_id
        st.session_state.messages = convo["chat_history"]
        st.session_state.title = convo["title"]

if st.sidebar.button("加载更多对话", use_container_width=True):
    st.session_state.page += 1
    history = get_history(st.session_state.page)
    print(history)
    if history is not None:
        for convo in history:
            convo_id = convo["conversation_id"]
            if convo_id not in st.session_state.history_list:
                st.session_state.history_list[convo_id] = convo
        # 重新排序并刷新页面
        st.session_state.history_list = OrderedDict(
            sorted(
                st.session_state.history_list.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True,
            )
        )
        st.rerun()
    else:
        st.session_state.page -= 1
        st.warning("没有更多对话记录了！")

messages_history = []

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"], avatar="☺️"):
            st.markdown(message["content"])
            messages_history.append(HumanMessage(message["content"]))
    else:
        with st.chat_message(message["role"], avatar="🤖"):
            st.markdown(message["content"])
            messages_history.append(AIMessage(message["content"]))

if prompt := st.chat_input("输入你的问题"):
    with st.chat_message("user", avatar="☺️"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        app = build_app()
        # response = generate(app, st.session_state.conversation_id, messages_history, prompt)

    except Exception as e:
        st.error(f"AI 生成响应失败: {e}")

    with st.chat_message('assistant', avatar='🤖'):
        # st.markdown(response['answer'])
        response = st.write_stream(
            generate(app, st.session_state.conversation_id, messages_history, prompt)
        )
    
    st.session_state.messages.append({'role': 'assistant', 'content': response})

    tokens_used = num_tokens_from_string(response)

    if st.session_state.title == "":
        st.session_state.title = generate_title(st.session_state.messages)

    # 创建消息
    conversation = {
        "conversation_id": st.session_state.conversation_id,
        "chat_history": st.session_state.messages,
        "tokens_used": tokens_used,
        "logs": st.session_state.logs,
        "title": st.session_state.title
    }

    # 发布消息到中间件
    publish_message(platform="log", message=conversation)

    # 延迟五秒
    time.sleep(2)
    st.rerun()
