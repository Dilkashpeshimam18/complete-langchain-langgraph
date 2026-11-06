from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class InterviewState(TypedDict):
    candidate_name: str
    question: str
    answer: str
    score: int

def ask_question(state: InterviewState):
    print("Asking interview question...")

    return {
        "question": "What is React?",
    }


def evaluate_answer(state: InterviewState):
    print("Evaluating candidate answer...")

    answer = state["answer"]

    score = 8 if len(answer) > 20 else 4

    return {
        "score": score
    }

builder = StateGraph(InterviewState)

builder.add_node("ask_question", ask_question)
builder.add_node("evaluate_answer", evaluate_answer)

builder.add_edge(START, "ask_question")
builder.add_edge("ask_question", "evaluate_answer")
builder.add_edge("evaluate_answer", END)

checkpointer=InMemorySaver()
graph=builder.compile(
    checkpointer=checkpointer
)

config={
    "configurable":{
        "thread_id":"candidate_101"
    }
}

result=graph.invoke({
"candidate_name": "Dilkash",
        "question": "",
        "answer": "React is a JavaScript library used to build user interfaces.",
        "score": 0
},
config=config
)

state = graph.get_state(config)

print(state.values)
print(state.config)
print(state.next)
print(result)