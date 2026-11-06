from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnablePassthrough, RunnableParallel,    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = StrOutputParser()

pdfLoader=PyPDFLoader(
    "data/resume.pdf"
)

documents=pdfLoader.load();
textSplitter= RecursiveCharacterTextSplitter(
     chunk_size=500,
    chunk_overlap=50
)

chunks=textSplitter.split_documents(documents)

llm=ChatOllama(
    model="llama3.2",
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )



prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful question-answering assistant.

        Answer using only the provided context.

        If the answer cannot be found in the context,
        say you don't know.

        Do not make up information.
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])




parallel = RunnableParallel(
    context=retriever | RunnableLambda(format_docs),
    question=RunnablePassthrough()
)


chain = (
    parallel
    | prompt
    | llm
    | StrOutputParser()
)

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a search query optimizer.

Rewrite the user's question into a concise search query
for a knowledge base.

Rules:
- Preserve the original meaning.
- Preserve important names, entities, dates and numbers.
- Do not answer the question.
- Remove unnecessary conversational wording.
- Return only the rewritten query.

User question:
{question}

Rewritten query:
""")

multi_query_prompt = ChatPromptTemplate.from_template("""
You are an AI search query generator.

Generate 3 different search queries based on the
user's original question.

Rules:
- Preserve the original meaning.
- Use different wording for each query.
- Do not answer the question.
- Return exactly 3 queries.
- Put each query on a separate line.
- Do not number the queries.

Original question:
{question}
""")

rewrite_chain = (
     multi_query_prompt
    | llm
    | StrOutputParser()
)
question = "What is the pdf about?"

rewritten_query=rewrite_chain.invoke(
    {
    "question": question
}
)

print("Original:", question)
print("Rewritten:", rewritten_query)

response = chain.invoke(rewritten_query)


print("\nANSWER:")

print(response)
