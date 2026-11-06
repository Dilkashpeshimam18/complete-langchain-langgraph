from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage


class AgentState(TypedDict):
    messages:list[BaseMessage]


@tool
def add(a:float,b:float)->float:
    """Add two number"""
    return a+b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second number."""

    if b == 0:
        raise ValueError("Cannot divide by zero")

    return a / b

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

tools=[add,multiply,divide]

llm_with_tools=llm.bind_tools(tools)

def chatbot(state:AgentState):
    response=llm_with_tools.invoke(
        state['messages']
    )
    return {
        "messages" :[response]
    }


builder=StateGraph(AgentState)

builder.add_node('chatbot', chatbot)
builder.add_node('tools',ToolNode(tools))

builder.add_edge(START,'chatbot')
builder.add_conditional_edges('chatbot', tools_condition)

builder.add_edge('tools','chatbot')

graph=builder.compile()

result=graph.invoke({
    "messages":[
        HumanMessage(
            content="What is 25 multiplied by 8, plus 50?"
        )
    ]
})

print("\nFinal Answer:")

print(result["messages"][-1].content)