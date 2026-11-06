from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class InterviewState(TypedDict):
    candidate_name: str
    evaluation: str
    final_report: str

def generate_evaluation(state: InterviewState):

    print("Generating candidate evaluation...")

    return {
        "evaluation": "Candidate has strong React skills but needs improvement in system design."
    }

def review_evaluation(state: InterviewState):

    decision = interrupt({
        "message": "Review candidate evaluation",
        "evaluation": state["evaluation"],
        "options": ["approve", "edit", "reject"]
    })

    if decision["action"] == "edit":
        return {
            "evaluation": decision["updated_evaluation"]
        }

    return {}

def publish_report(state: InterviewState):

    return {
        "final_report": state["evaluation"]
    }

builder = StateGraph(InterviewState)

builder.add_node("generate_evaluation", generate_evaluation)
builder.add_node("review_evaluation", review_evaluation)
builder.add_node("publish_report", publish_report)

builder.add_edge(START, "generate_evaluation")
builder.add_edge("generate_evaluation", "review_evaluation")
builder.add_edge("review_evaluation", "publish_report")
builder.add_edge("publish_report", END)

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)
config = {
    "configurable": {
        "thread_id": "candidate_101"
    }
}

result = graph.invoke(
    {
        "candidate_name": "Dilkash",
        "evaluation": "",
        "final_report": ""
    },
    config=config
)

print(result)
print(result["__interrupt__"])
result = graph.invoke(
    Command(
        resume={
            "action": "approve"
        }
    ),
    config=config
)

print(result)