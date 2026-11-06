
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# 1. Original documents
parent_docs = [
    Document(
        page_content=(
            "RAG combines information retrieval with language generation. "
            "It retrieves relevant external documents and uses them as "
            "context to help produce grounded answers."
        ),
        metadata={"topic": "RAG"}
    ),
    Document(
        page_content=(
            "Fine-tuning adapts a pretrained model using additional "
            "training examples. It can improve task-specific behavior "
            "and response style."
        ),
        metadata={"topic": "Fine-tuning"}
    )
]

# 2. Configure models
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# 3. Generate summaries
summary_prompt = ChatPromptTemplate.from_template("""
Summarize the following document in two concise sentences.
Preserve its main concepts.

Document:
{document}
""")

summary_chain = summary_prompt | llm | StrOutputParser()

# 4. Generate hypothetical questions
question_prompt = ChatPromptTemplate.from_template("""
Generate three questions that this document can answer.
Return one question per line, without numbering.

Document:
{document}
""")

question_chain = question_prompt | llm | StrOutputParser()

# 5. Create a vector store for representations
vectorstore = Chroma(
    collection_name="multi_vector_demo",
    embedding_function=embeddings
)

# Store original documents by ID
docstore = {
    str(uuid4()): doc for doc in []
}

# 6. Index multiple representations
parent_store = {}

for parent_doc in parent_docs:
    parent_id = str(uuid4())
    parent_store[parent_id] = parent_doc

    summary = summary_chain.invoke({
        "document": parent_doc.page_content
    })

    generated_questions = question_chain.invoke({
        "document": parent_doc.page_content
    })

    representations = [
        ("original", parent_doc.page_content),
        ("summary", summary)
    ]

    for generated_question in generated_questions.splitlines():
        generated_question = generated_question.strip()
        generated_question = generated_question.lstrip("0123456789.-) ")

        if generated_question:
            representations.append(
                ("question", generated_question)
            )

    vector_docs = []

    for representation_type, text in representations:
        vector_docs.append(
            Document(
                page_content=text,
                metadata={
                    "parent_id": parent_id,
                    "representation_type": representation_type,
                    **parent_doc.metadata
                }
            )
        )

    vectorstore.add_documents(vector_docs)

# 7. Search across all representations
query = "How can an LLM reduce hallucinations?"

matches = vectorstore.similarity_search(
    query,
    k=5
)

# 8. Map results to unique original documents
results = []
seen_parent_ids = set()

for match in matches:
    parent_id = match.metadata["parent_id"]

    if parent_id not in seen_parent_ids:
        seen_parent_ids.add(parent_id)
        results.append(parent_store[parent_id])

# 9. Display original documents
for doc in results:
    print("\n--- ORIGINAL DOCUMENT ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
