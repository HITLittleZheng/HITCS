from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, AIMessageChunk
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing import Dict, Any
import tiktoken


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    num_tokens = len(encoding.encode(string))
    return num_tokens


class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    answer: str
    metadata: Dict[str, Any]

load_dotenv()
ZHIPU_API_KEY = os.getenv("SECRET_KEY")

llm = ChatZhipuAI(
    api_key=ZHIPU_API_KEY,
    temperature=0.5,
    model="glm-4-flash",
    streaming=True
)
def build_app():
    system_prompt = "You're an AI assistant. You're helping a user with a question. "
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
            "metadata": response.response_metadata,
        }

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "llm")
    workflow.add_node("llm", call_model)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def generate(app, conversation_id, messages_history, input):
    config = {"configurable": {"thread_id": conversation_id}}
    for msg,metadata in app.stream(
        {"input": input, "chat_history": messages_history},
        config=config,stream_mode='messages'
    ):
        if isinstance(msg, AIMessageChunk):
            yield msg.content


def generate_title(chat_history):
    title_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarizing the below chat content as a short title, don't include any punctuation."),
            ("human", "{input}"),
        ]
    )
    chain = title_prompt | llm 
    response = chain.invoke({"input": chat_history})
    print(response.content)
    return response.content
