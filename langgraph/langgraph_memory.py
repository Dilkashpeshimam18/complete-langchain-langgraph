from typing import TypedDict,Annotated

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore

class ChatState(TypedDict):
    messages: Annotated[list,add_messages]

def chatbot(state: ChatState):

    last_message = state["messages"][-1]

    user_input = last_message.content
    response = AIMessage(
        content=f"You said: {user_input}"
    )

    return {
        "messages": [response]
    }

builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Short - term Memory
checkpointer = InMemorySaver()


graph = builder.compile(
    checkpointer=checkpointer
)

config = {
    "configurable": {
        "thread_id": "user_101"
    }
}

response1 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="My name is Dilkash."
            )
        ]
    },
    config=config
)

print(response1["messages"])

response2 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is my name?"
            )
        ]
    },
    config=config
)

print(response2["messages"])

# Long - term Memory

store=InMemoryStore()

namespace=("users",'user_101')

store.put(
  namespace,
  "profile",
  {
      "name": "Dilkash",
        "skills": ["React", "Node.js", "Python"],
        "experience": "3 years"
  }
)

memory=store.get(
    namespace,
    "profile"
)

print(memory.value)