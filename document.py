from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_community.document_loaders import JSONLoader

from langchain_text_splitters import RecursiveJsonSplitter

load_dotenv()

textloader= TextLoader("data/sample.txt")

pdfLoader=PyPDFLoader("data/resume.pdf")


jsonLoader = JSONLoader(
    file_path="data/company.json",
    jq_schema=".",
        text_content=False

)


documents= textloader.load()

pdfDocument=pdfLoader.load()

jsonDocuments = jsonLoader.load()



splitter= RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

characterSplitter=CharacterTextSplitter(
    separator='\n\n',
    chunk_size=200,
    chunk_overlap=20
)

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

htmlSplitter=HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

jsonSplitter = RecursiveJsonSplitter(
    max_chunk_size=300
)

chunks=splitter.split_documents(documents)
pdfChunks=splitter.split_documents(pdfDocument)
characterChunk=characterSplitter.split_documents(documents)
htmlChunks=htmlSplitter.split_text_from_file("data/sample.html")

print(f"Number of chunks: {len(chunks)}")
print(f"Number of Pdf chunks: {len(pdfChunks)}")



for i, chunk in enumerate(chunks):

    print(f"\n--- CHUNK {i} ---")

    print(chunk.page_content)

    print("\nMetadata:")

    print(chunk.metadata)


for i, chunk in enumerate(pdfChunks[:5]):

    print(f"\n===== CHUNK {i} =====")

    print(chunk.page_content)

    print("\nMetadata:")

    print(chunk.metadata)


for i, chunk in enumerate(htmlChunks):

    print(f"\n--- CHUNK {i + 1} ---")

    print("CONTENT:")
    print(chunk.page_content)

    print("METADATA:")
    print(chunk.metadata)



for document in jsonDocuments:
    print(document.page_content)