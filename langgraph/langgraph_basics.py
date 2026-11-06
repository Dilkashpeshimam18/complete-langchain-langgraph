from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class InterviewState(TypedDict):
    question:str
    answer:str
    feedback:str

def generateAnswer(state:InterviewState):
    answer = (
        "A REST API allows applications to communicate "
        "using HTTP methods such as GET and POST."
    )
    return {"answer":answer}

def evaluate_answer(state: InterviewState):
    answer=state["answer"]
    if "HTTP" in answer and "GET" in answer:
        feedback = "Good answer: key concepts are included."
    else:
        feedback = "Improve the answer with relevant examples."

    return {"feedback": feedback}

builder=StateGraph(InterviewState)

builder.add_node("generate",generateAnswer)
builder.add_node("evaluate",evaluate_answer)

builder.add_edge(START,'generate')
builder.add_edge('generate','evaluate')
builder.add_edge('evaluate',END)

graph=builder.compile()

result=graph.invoke({
    "question": "What is a REST API?",
    "answer": "",
    "feedback": ""
})


print("Question:", result["question"])
print("Answer:", result["answer"])
print("Feedback:", result["feedback"])