
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# 1. Define Pydantic schema
class InterviewEvaluation(BaseModel):

    score: int = Field(
        description="Overall score out of 10",
        ge=0,
        le=10
    )

    feedback: str = Field(
        description="Detailed evaluation feedback"
    )

    strengths: list[str] = Field(
        description="Candidate's strong points"
    )

    weaknesses: list[str] = Field(
        description="Areas needing improvement"
    )

    recommendation: str = Field(
        description="Recommended next step"
    )


# 2. Define LangGraph state
class InterviewState(TypedDict):

    question: str
    candidate_answer: str
    evaluation: dict


# 3. Initialize LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

structured_llm = llm.with_structured_output(
    InterviewEvaluation
)


# 4. Create prompt
prompt = ChatPromptTemplate.from_template("""
You are an experienced technical interviewer.

Evaluate the candidate's answer.

Question:
{question}

Candidate Answer:
{candidate_answer}

Evaluate based on:
1. Technical accuracy
2. Clarity
3. Completeness

Provide:
- Score out of 10
- Detailed feedback
- Strengths
- Weaknesses
- Recommendation
""")


# 5. Evaluation node
def evaluate_answer(state: InterviewState):

    chain = prompt | structured_llm

    response = chain.invoke({
        "question": state["question"],
        "candidate_answer": state["candidate_answer"]
    })

    print("\nLLM Evaluation:")

    print("Score:", response.score)
    print("Feedback:", response.feedback)
    print("Strengths:", response.strengths)
    print("Weaknesses:", response.weaknesses)
    print("Recommendation:", response.recommendation)

    return {
        "evaluation": response.model_dump()
    }


# 6. Build graph
builder = StateGraph(InterviewState)

builder.add_node("evaluate", evaluate_answer)

builder.add_edge(START, "evaluate")
builder.add_edge("evaluate", END)


# 7. Compile graph
graph = builder.compile()


# 8. Execute graph
result = graph.invoke({

    "question": "What is a REST API?",

    "candidate_answer": """
    A REST API allows applications to communicate
    using HTTP methods such as GET and POST.
    """,

    "evaluation": {}
})


# 9. Display final result
print("\n========== FINAL EVALUATION ==========")

evaluation = result["evaluation"]

print("Score:", evaluation["score"])
print("Feedback:", evaluation["feedback"])
print("Strengths:", evaluation["strengths"])
print("Weaknesses:", evaluation["weaknesses"])
print("Recommendation:", evaluation["recommendation"])
