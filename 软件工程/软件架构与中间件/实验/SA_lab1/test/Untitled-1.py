from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing import Dict, Any


class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    answer: str
    metadata: Dict[str, Any]


load_dotenv()

def build_app():
llm = ChatZhipuAI(
    api_key=ZHIPU_API_KEY,
    temperature=0.5,
    model="glm-4-flash",
)


system_prompt = "你是一个为了软件架构而生的AI助手，辅助进行软件架构设计，你的名字是HIT软件架构小助手"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


chain = prompt | llm


def call_model(state: State):
    response = chain.invoke(state)

    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response.content),
        ],
        "answer": response.content,
        "metadata": response.response_metadata
    }


# Define a new graph
workflow = StateGraph(state_schema=State)

# Define the (single) node in the graph
workflow.add_edge(START, "llm")
workflow.add_node("llm", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "abc123"}}
query = "你好，我叫Bob"


result = app.invoke({"input": query}, config=config)

