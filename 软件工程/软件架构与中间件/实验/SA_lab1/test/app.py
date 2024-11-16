import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import uuid
import socket
import json
import redis
import os
from dotenv import load_dotenv
from chain import build_app, generate
load_dotenv()

# Redis 客户端配置（用于即时更新历史）
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# # LangChain LLM 初始化
# ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')

# # LangChain LLM 初始化
# llm = ChatZhipuAI(
#     api_key=ZHIPU_API_KEY,
#     temperature=0.5,
#     model="glm-4-flash",
# )


def get_history():
    try:
        keys = redis_client.keys("conversation:*")
        history = []
        for key in keys:
            convo = redis_client.hgetall(key)
            history.append(
                {
                    "conversation_id": key.decode().split(":")[1],
                    "messages": json.loads(convo.get(b"messages", b"[]").decode()),
                }
            )
        # 按时间排序或其他逻辑
        return history
    except Exception as e:
        st.sidebar.error(f"Error fetching history: {e}")
        return []


def publish_message(message, host="localhost", port=9999):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        # Send the message as JSON string
        print(("PUBLISHER\n"+ json.dumps(message)+'\n'))
        client.sendall(("PUBLISHER\n"+ json.dumps(message)+'\n').encode())
        client.close()
        print("Published message to Message Broker.")
    except Exception as e:
        st.error(f"Error publishing message: {e}")

st.title("AI 问答系统")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "logs" not in st.session_state:
    st.session_state.logs = []

# 侧边栏显示历史对话
st.sidebar.header("历史对话")

history = get_history()
for convo in history:
    if st.sidebar.button(f"对话 {convo['conversation_id']}", key=convo['conversation_id'],use_container_width=True):
        st.session_state.conversation_id = convo['conversation_id']
        st.session_state.messages = []
        for messages in convo['messages']:
            st.session_state.messages.append(messages)

if st.sidebar.button("新建对话", use_container_width=True):
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []

messages_history=[]

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
        response = generate(app, st.session_state.conversation_id, messages_history, prompt)

    except Exception as e:
        st.error(f"AI 生成响应失败: {e}")

    with st.chat_message('assistant', avatar='🤖'):
        st.markdown(response['answer'])
    st.session_state.messages.append({'role': 'assistant', 'content': response['answer']})

    # 计算 token 使用量（示例，需根据实际情况调整）
    tokens_used = int(response['metadata']['token_usage']['total_tokens'])

    # 创建消息
    conversation = {
        "conversation_id": st.session_state.conversation_id,
        "messages": st.session_state.messages,
        "tokens_used": tokens_used,
        "logs":st.session_state.logs
    }

    
    # 发布消息到中间件
    publish_message(conversation)

    st.rerun()
