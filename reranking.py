
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

chunks = [
    Document(
        page_content="Employees receive 24 days of annual leave.",
        metadata={
            "department": "HR",
            "year": 2025,
            "source": "leave_policy.pdf"
        }
    ),
    Document(
        page_content="Employees can claim approved travel expenses.",
        metadata={
            "department": "Finance",
            "year": 2025,
            "source": "expenses_policy.pdf"
        }
    ),
    Document(
        page_content="Engineers use code reviews before merging changes.",
        metadata={
            "department": "Engineering",
            "year": 2025,
            "source": "engineering_guide.pdf"
        }
    ),
    Document(
        page_content="The HR team provides 30 days of parental leave.",
        metadata={
            "department": "HR",
            "year": 2024,
            "source": "parental_leave.pdf"
        }
    )
]

# 1. Create vector store from your chunks
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="reranking_demo"
)

# 2. Retrieve candidate documents
question = "What are the benefits of RAG?"

candidate_docs=vectorstore.similarity_search(
    question,
    k=10
)

reranker=CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# 4. Prepare query-document pairs
pairs = [
    (question, doc.page_content)
    for doc in candidate_docs
]

# 5. Score each document for relevance
scores = reranker.predict(pairs)

# 6. Sort documents by reranker score
ranked_results = sorted(
    zip(candidate_docs, scores),
    key=lambda item: float(item[1]),
    reverse=True
)

# 7. Display the reranked documents
for rank, (doc, score) in enumerate(
    ranked_results[:3], start=1
):
    print(f"\nRank: {rank}")
    print(f"Score: {float(score):.4f}")
    print(doc.page_content)
