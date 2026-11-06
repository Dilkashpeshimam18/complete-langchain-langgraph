
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document


documents = [
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

# 1. Create embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 2. Create a vector store from your document chunks
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="hybrid_demo"
)

vector_retriever=vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

bm25_retriever=BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = 3

hybrid_retriever=EnsembleRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    weights=[0.5, 0.5]
)

# 6. Run hybrid search
question = "What are the employee leave?"

results = hybrid_retriever.invoke(question)

# 7. Print retrieved documents
for i, doc in enumerate(results, start=1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
