from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

texts = [
    "Python is a programming language.",
    "React is used to build user interfaces.",
    "RAG allows LLMs to retrieve information from documents.",
    "Mumbai is a city in India."
]


vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


print("Chroma vector store created!")

query = "How does an LLM retrieve information?"


results = vectorstore.similarity_search(
    query,
    k=2
)


for i, document in enumerate(results):

    print(f"\n--- RESULT {i + 1} ---")
    print(document.page_content)