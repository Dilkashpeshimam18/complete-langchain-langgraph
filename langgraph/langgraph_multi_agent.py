from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class InterviewState(TypedDict):
    resume: str
    job_description: str
    resume_analysis: str
    skill_gap: str
    interview_questions: str


llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

def resume_agent(state: InterviewState):

    prompt=ChatPromptTemplate([
        (
            "system",
             "You are an expert technical resume analyst. "
            "Identify the candidate's skills, experience, "
            "strengths and technical background. "
            "Do not invent missing experience."
        ),
        (
            "human",
             "Analyze this resume:\n{resume}"
        )
    ])

    chain=prompt | llm
    response=chain.invoke({
        "resume":state['resume']
    })

    return {
        "resume_analysis":response.content
    }

def skill_gap_agent(state: InterviewState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a technical skill gap analyst. "
            "Compare the candidate's demonstrated skills "
            "with the job requirements. "
            "Identify matching skills, missing skills "
            "and preparation priorities."
        ),
        (
            "human",
            "Resume analysis:\n{resume_analysis}\n\n"
            "Job description:\n{job_description}"
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
        "resume_analysis": state["resume_analysis"],
        "job_description": state["job_description"]
    })

    return {
        "skill_gap": response.content
    }

def interview_agent(state: InterviewState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an experienced technical interviewer. "
            "Generate exactly five relevant interview questions. "
            "Include conceptual and practical questions."
        ),
        (
            "human",
            "Candidate analysis:\n{resume_analysis}\n\n"
            "Skill gap analysis:\n{skill_gap}"
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
        "resume_analysis": state["resume_analysis"],
        "skill_gap": state["skill_gap"]
    })

    return {
        "interview_questions": response.content
    }

builder = StateGraph(InterviewState)

builder.add_node("resume_agent", resume_agent)
builder.add_node("skill_gap_agent", skill_gap_agent)
builder.add_node("interview_agent", interview_agent)

builder.add_edge(START, "resume_agent")
builder.add_edge("resume_agent", "skill_gap_agent")
builder.add_edge("skill_gap_agent", "interview_agent")
builder.add_edge("interview_agent", END)

graph = builder.compile()

result = graph.invoke({
    "resume": """
    Full Stack Developer with 3 years of experience.

    Skills:
    React, Next.js, Node.js, Express,
    Python, PostgreSQL, MongoDB, AWS.
    """,

    "job_description": """
    Looking for an AI Engineer with experience in:
    Python, FastAPI, LangChain, LangGraph,
    RAG, vector databases, Docker and AWS.
    """,

    "resume_analysis": "",
    "skill_gap": "",
    "interview_questions": ""
})

print("\n===== RESUME ANALYSIS =====")
print(result["resume_analysis"])

print("\n===== SKILL GAP ANALYSIS =====")
print(result["skill_gap"])

print("\n===== INTERVIEW QUESTIONS =====")
print(result["interview_questions"])