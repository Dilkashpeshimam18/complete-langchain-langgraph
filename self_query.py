
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_classic.retrievers.self_query.base import (
    SelfQueryRetriever
)
from langchain_classic.chains.query_constructor.base import (
    AttributeInfo
)

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

# 1. Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 2. Vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="self_query_demo"
)

# 3. LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# 4. Describe the metadata fields available for filtering
metadata_field_info = [
    AttributeInfo(
        name="department",
        description="Department responsible for the document, such as HR, Finance, or Engineering",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="Year the document was published",
        type="integer"
    ),
    AttributeInfo(
        name="source",
        description="Filename of the source document",
        type="string"
    )
]

# 5. Describe the document content
document_content_description = (
    "Company policies and internal guides covering HR, "
    "employee leave, finance expenses, and engineering practices"
)

# 6. Create the self-query retriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)

# 7. Ask a natural-language question
question = "Find HR policies published after 2024"
results = retriever.invoke(question)

# 8. Print results
for doc in results:
    print("\n--- DOCUMENT ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
