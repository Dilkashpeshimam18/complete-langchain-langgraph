from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small"
)

text="I love embeddings"

texts = [
    "I love programming",
    "I enjoy coding",
    "I like eating pizza"
]

vector= embeddings.embed_query(text)

vectors=embeddings.embed_documents(texts)

similarity=cosine_similarity(
    [vectors[0]],
    [vectors[2]]
)

print("Similarity:")
print(similarity[0][0])