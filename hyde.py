
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma


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

# 2. LLM for generating a hypothetical document
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# 3. HyDE prompt
hyde_prompt = ChatPromptTemplate.from_template("""
Write a hypothetical answer to the user's question.

It is used only to improve semantic retrieval.
It does not need to be factually correct.

Question:
{question}

Hypothetical answer:
""")

hyde_chain = hyde_prompt | llm | StrOutputParser()

# 4. Original question
question = "What are the benefits of RAG?"

# 5. Generate hypothetical document
hypothetical_document = hyde_chain.invoke({
    "question": question
})

print("\nHYPOTHETICAL DOCUMENT:")
print(hypothetical_document)

# 6. Create the vector store from real chunks ONCE.
# Make sure `chunks` is already defined.
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 7. Search using the hypothetical document.
# Chroma automatically embeds this query.
documents = vectorstore.similarity_search(
    query=hypothetical_document,
    k=3
)

# 8. Display retrieved real documents
print("\nRETRIEVED DOCUMENTS:")

for doc in documents:
    print("\n--- DOCUMENT ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
