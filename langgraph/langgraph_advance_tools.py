from typing import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage


class AgentState(TypedDict):
    messages: list[BaseMessage]

@tool
def analyze_resume(skills: str) -> str:
    """Analyze candidate skills and provide a skill assessment."""

    return f"Candidate skills: {skills}. Resume analysis completed."


@tool
def calculate_score(score: int, total: int) -> str:
    """Calculate interview percentage from score and total marks."""

    if total <= 0:
        raise ValueError("Total marks must be greater than zero.")

    percentage = (score / total) * 100

    return f"Interview percentage: {percentage:.2f}%"


@tool
def generate_learning_plan(skill: str) -> str:
    """Generate a learning plan for a specific technical skill."""

    return f"Learning plan generated for {skill}."

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

tools = [
    analyze_resume,
    calculate_score,
    generate_learning_plan
]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state:AgentState):
    response=llm_with_tools.invoke(
        state["messages"]
    )
    return {
        "messages" : [response]
    }

builder=StateGraph(AgentState)

builder.add_node("chatbot", chatbot)

builder.add_node('tools',
 ToolNode(tools,handle_tool_errors=True)
)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges('chatbot',tools_condition)

builder.add_edge('tools','chatbot')

graph=builder.compile()

response=graph.invoke({
    "messages":[HumanMessage(
        content="Analyze my resume. My skills are React, Node.js, Python and PostgreSQL. Identify my skill profile."
    )]
})

print("\nFINAL RESPONSE:")
print(response["messages"][-1].content)