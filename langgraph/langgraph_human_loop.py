from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt,Command

class AgentState(TypedDict):
    request: str
    proposed_action: str
    approval: str
    result: str

def agent(state: AgentState):

    print("Agent is analyzing request...")

    return {
        "proposed_action": "Send interview evaluation report to recruiter"
    }


def human_approval(state:AgentState):
    decision=interrupt({
      "message":'Please approve this action',
        "action": state['proposed_action']
    })

    return {
        "approval": decision
    }

def execute_action(state: AgentState):

    if state["approval"] == "approve":

        return {
            "result": "Action approved and executed!"
        }

    return {
        "result": "Action rejected. Nothing was executed."
    }

builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("human_approval", human_approval)
builder.add_node("execute_action", execute_action)

builder.add_edge(START, "agent")
builder.add_edge("agent", "human_approval")
builder.add_edge("human_approval", "execute_action")
builder.add_edge("execute_action", END)

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)

config = {
    "configurable": {
        "thread_id": "interview_101"
    }
}

result = graph.invoke(
    {
        "request": "Send candidate evaluation report",
        "proposed_action": "",
        "approval": "",
        "result": ""
    },
    config=config
)

response=graph.invoke(
    Command(resume='approve'),
    config=config
)
print(result)
print(result["__interrupt__"])
print(response)
