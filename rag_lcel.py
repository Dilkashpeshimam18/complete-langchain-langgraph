
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


prompt=ChatPromptTemplate(
    [
        (
 "system",
        """You are a helpful question-answering assistant.

Answer the question using only the provided context.

If the answer is not present in the context,
say that you don't know."""
        ),
        (
"human",
        """Context:

{context}

Question:

{question}"""
        )
    ]
)


llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

chain= prompt | llm | format_docs

response=chain.invoke("Hi, How are you?")