from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small"
)

texts = [
    "Python is a programming language.",
    "React is used to build user interfaces.",
    "RAG allows LLMs to retrieve information from external documents.",
    "Mumbai is a city in India."
]

vectorstore= FAISS.from_texts(
    texts,
    embeddings
)

retriever=vectorstore.as_retriever(search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    });

query = "How does RAG retrieve information?"

results=retriever.invoke(query)

for i, document in enumerate(results):

    print(f"\n--- RESULT {i + 1} ---")
    print(document.page_content)