from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define shared state
class InterviewState(TypedDict):
    question: str
    answer: str
    score: int
    feedback: str
    attempts: int


# 2. Generate initial answer
def generate_answer(state: InterviewState):

    print("\nGenerating answer...")

    return {
        "answer": "REST API is used for communication.",
        "attempts": state["attempts"] + 1
    }


# 3. Evaluate answer
def evaluate_answer(state: InterviewState):

    answer = state["answer"]

    if "HTTP" in answer and "GET" in answer:
        score = 8
        feedback = "Good answer. Important concepts covered."

    elif "communication" in answer:
        score = 4
        feedback = "Explain HTTP methods and their purpose."

    else:
        score = 2
        feedback = "Your answer needs more technical detail."

    print("Score:", score)

    return {
        "score": score,
        "feedback": feedback
    }


# 4. Improve answer
def improve_answer(state: InterviewState):

    print("Improving answer...")

    if state["attempts"] == 1:
        answer = (
            "REST APIs allow communication between applications "
            "using HTTP methods such as GET and POST."
        )
    else:
        answer = (
            "REST APIs allow client-server communication "
            "using HTTP methods such as GET, POST, PUT and DELETE. "
            "They commonly exchange data using JSON."
        )

    return {"answer": answer}


# 5. Routing logic
def route_answer(state: InterviewState):

    if state["score"] >= 7:
        return "success"

    if state["attempts"] >= 3:
        return "max_attempts"

    return "retry"

builder=StateGraph(InterviewState)
builder.add_node("generate",generate_answer)
builder.add_node("evaluate",evaluate_answer)
builder.add_node('improve',improve_answer)

builder.add_edge(START,'generate')
builder.add_edge('generate','evaluate')

builder.add_conditional_edges(
    "evaluate",
    route_answer,
    {
        "success": END,
        "retry": "improve",
        "max_attempts": END
    }

)

builder.add_edge("improve", "evaluate")

graph=builder.compile()

# 8. Execute
result = graph.invoke(
    {
        "question": "What is a REST API?",
        "answer": "",
        "score": 0,
        "feedback": "",
        "attempts": 0
    },
    config={"recursion_limit": 10}
)


print("\n========== FINAL RESULT ==========")
print("Question:", result["question"])
print("Answer:", result["answer"])
print("Score:", result["score"])
print("Feedback:", result["feedback"])
print("Attempts:", result["attempts"])