from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

class InterviewEvaluation(BaseModel):
    score:int
    strength: list[str]
    weakness: list[str]
    recommendation:str

class CandidateAnalysis(BaseModel):
    name: str
    experience: int
    skills: list[str]


parser=PydanticOutputParser(
    pydantic_object=CandidateAnalysis
)

parser1=PydanticOutputParser(
    pydantic_object=InterviewEvaluation
)

text = """
{
    "name": "Dilkash",
    "experience": 3,
    "skills": ["React", "Node.js", "Python"]
}
"""

text1 = """
{
    "score": 80,
    "strength": ["React", "Node.js", "Python"],
    "weakness": ["System Design", "Kubernetes"],
    "recommendation": "Improve system design and DevOps knowledge"
}
"""

result1 = parser1.invoke(text1)

print(result1)
result=parser.invoke(text)



print(result)
