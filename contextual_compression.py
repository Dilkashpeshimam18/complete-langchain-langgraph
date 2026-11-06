
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load your PDF
loader = PyPDFLoader("data/resume.pdf")
documents = loader.load()

# 2. Split the PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# 1. Embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 2. Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="compression_demo"
)

# 3. Base retriever
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# 4. LLM used to extract relevant information
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# 5. Create the compressor
compressor = LLMChainExtractor.from_llm(llm)

# 6. Wrap the retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 7. Retrieve compressed documents
question = "What are the benefits of RAG?"

compressed_docs = compression_retriever.invoke(question)

# 8. Display results
for i, doc in enumerate(compressed_docs, start=1):
    print(f"\n--- Compressed Document {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
