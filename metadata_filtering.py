
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

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

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="metadata_demo"
)

results=vectorstore.similarity_search(
    query="What leave benefits are available?",
    k=3,
    filter={
        "$and": [
            {"department": "HR"},
            {"year": {"$gte": 2025}}
        ]
        
    }
)

for doc in results:
    print("\nDocument:", doc.page_content)
    print("Metadata:", doc.metadata)