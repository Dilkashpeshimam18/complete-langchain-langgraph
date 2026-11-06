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
    "RAG allows LLMs to retrieve information.",
    "Mumbai is a city in India.",
]



vectorStore=FAISS.from_texts(
    texts,
    embedding=embeddings
)

print("Vector store created!")


query="How can an LLM retrieve information?"

results= vectorStore.similarity_search(
    query,
    k=2
)

for i, result in enumerate(results):

    print(f"\n--- RESULT {i + 1} ---")
    print(result.page_content)