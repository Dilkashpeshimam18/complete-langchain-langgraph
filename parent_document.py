
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Create sample parent documents
parent_docs = [
    Document(
        page_content=(
            "RAG connects a language model to external knowledge. "
            "It can reduce unsupported answers by grounding responses "
            "in retrieved information. It also allows information to "
            "be updated without retraining the language model. "
            "RAG can provide source references and domain-specific knowledge."
        ),
        metadata={"topic": "RAG"}
    ),
    Document(
        page_content=(
            "Fine-tuning trains a model further on selected examples. "
            "It can help adapt model behavior, style, and task performance. "
            "However, updating factual knowledge may require additional "
            "training and careful evaluation."
        ),
        metadata={"topic": "Fine-tuning"}
    )
]

# 2. Create small child chunks
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# 3. Create larger parent sections
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0
)

# 4. Create vector store and parent document store
vectorstore = Chroma(
    collection_name="parent_child_demo",
    embedding_function=OllamaEmbeddings(
        model="nomic-embed-text"
    )
)

docstore = InMemoryStore()

# 5. Configure ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# 6. Index the original documents
retriever.add_documents(parent_docs)

# 7. Search
question = "How does RAG improve reliability?"
results = retriever.invoke(question)

# 8. Display returned parent documents
for i, doc in enumerate(results, start=1):
    print(f"\n--- Parent Document {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
