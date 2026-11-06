
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ragas import EvaluationDataset, SingleTurnSample

# 1. Sample knowledge base
chunks = [
    Document(
        page_content="The candidate has 3 years of full-stack experience.",
        metadata={"source": "resume.pdf"}
    ),
    Document(
        page_content="The candidate has experience with Node.js and Python.",
        metadata={"source": "resume.pdf"}
    ),
    Document(
        page_content="The candidate has experience with AWS.",
        metadata={"source": "resume.pdf"}
    )
]

# 2. Create vector store
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_evaluation_demo"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 3. Configure LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context.
If the answer is missing, say you don't know.

Context:
{context}

Question:
{question}
""")

answer_chain = prompt | llm | StrOutputParser()

# 4. Evaluation test cases
test_cases = [
    {
        "question": "How many years of experience does the candidate have?",
        "expected": "The candidate has 3 years of full-stack experience."
    },
    {
        "question": "Which backend technologies are listed?",
        "expected": "The candidate has experience with Node.js and Python."
    },
    {
        "question": "Does the candidate have AWS experience?",
        "expected": "Yes, the candidate has experience with AWS."
    }
]

# 5. Run RAG and collect real outputs for evaluation
samples = []

for test in test_cases:
    question = test["question"]
    expected = test["expected"]

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    answer = answer_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nQUESTION:", question)
    print("EXPECTED:", expected)
    print("ACTUAL:", answer)
    print("RETRIEVED DOCUMENTS:", len(docs))
    print("-" * 50)

    samples.append(
        SingleTurnSample(
            user_input=question,
            retrieved_contexts=[
                doc.page_content for doc in docs
            ],
            response=answer,
            reference=expected
        )
    )

# 6. Build RAGAS evaluation dataset
dataset = EvaluationDataset(
    samples=samples
)

print("\nEvaluation samples collected:", len(dataset.samples))
