from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

webLoader=WebBaseLoader(
    "https://docs.langchain.com/oss/python/langchain/quickstart"
)

documents=webLoader.load();
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
question = "How to build an Agent?"



results = retriever.invoke(question)

for i, document in enumerate(results):

    print(f"\n--- RESULT {i + 1} ---")

    print(document.page_content)

    print("\nMETADATA:")
    print(document.metadata)

context = "\n\n".join(
    document.page_content
    for document in results
)
prompt = f"""
Answer the question using only the context provided below.

If the answer is not available in the context,
say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""


response = llm.invoke(prompt)


print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(response.content)

