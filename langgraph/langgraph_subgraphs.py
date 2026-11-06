from typing import TypedDict

from langgraph.graph import StateGraph, START, END

class InterviewState(TypedDict):
    candidate_name: str
    resume: str
    skills: str
    interview_answer: str
    evaluation: str

def analyze_resume(state: InterviewState):

    print("Analyzing resume...")

    return {
        "skills": "React, Node.js, Python, PostgreSQL"
    }

resume_builder = StateGraph(InterviewState)

resume_builder.add_node("analyze_resume", analyze_resume)

resume_builder.add_edge(START, "analyze_resume")
resume_builder.add_edge("analyze_resume", END)

resume_graph = resume_builder.compile()

def evaluate_interview(state: InterviewState):

    print("Evaluating interview answer...")

    answer = state["interview_answer"]

    if len(answer) > 30:
        evaluation = "Good answer. Candidate demonstrates understanding."
    else:
        evaluation = "Answer needs more technical depth."

    return {
        "evaluation": evaluation
    }

evaluation_builder = StateGraph(InterviewState)

evaluation_builder.add_node(
    "evaluate_interview",
    evaluate_interview
)

evaluation_builder.add_edge(
    START,
    "evaluate_interview"
)

evaluation_builder.add_edge(
    "evaluate_interview",
    END
)

evaluation_graph = evaluation_builder.compile()

parent_builder=StateGraph(InterviewState)

parent_builder.add_node('resume_analysis',resume_graph)
parent_builder.add_node('interview_evaluation',evaluation_graph)

parent_builder.add_edge(START,'resume_analysis')

parent_builder.add_edge('resume_analysis','interview_evaluation')
parent_builder.add_edge('interview_evaluation',END)

parent_graph=parent_builder.compile()

result = parent_graph.invoke({
    "candidate_name": "Dilkash",
    "resume": "Full-stack developer with 3 years of experience",
    "skills": "",
    "interview_answer": "React is a JavaScript library for building user interfaces using reusable components.",
    "evaluation": ""
})

print("\nFinal Result:")
print(result)