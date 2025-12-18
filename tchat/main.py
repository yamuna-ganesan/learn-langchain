from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}


def get_session_history(session_id):
    if session_id in store:
        return store[session_id]

    store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


model = ChatOllama(model="llama3:8b")

prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful assistant who speaks minimally"),
        ("placeholder", "{history}"),
        ("human", "{content}"),
    ]
)

chat_chain = prompt | model | StrOutputParser()
chat_with_history = RunnableWithMessageHistory(
    runnable=chat_chain,
    get_session_history=get_session_history,
    input_messages_key="content",
    history_messages_key="history",
)

while True:
    human_msg = input(">> ")
    ai_msg = chat_with_history.invoke(
        input={"content": human_msg},
        config={"configurable": {"session_id": "bill-gates"}},
    )
    print(ai_msg)

    chat_history = get_session_history("bill-gates")
    print(chat_history)
