import os
import time
import requests
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "pdfquora"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def download_pdf(url):
    print(f"Downloading PDF from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    print("✅ PDF downloaded successfully")
    return response.content


def split_pdf_chunks(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )
        chunks = splitter.split_documents(docs)
    finally:
        os.remove(tmp_path)

    print(f"✅ Split PDF into {len(chunks)} chunks")
    return chunks


def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(INDEX_NAME)
    print(f"✅ Initialized Pinecone index '{INDEX_NAME}'")
    return index


def embed_and_store(chunks, index):
    print("Embedding chunks and storing in Pinecone...")
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embedder,
        namespace="pdf",
        text_key="page_content"
    )

    vector_store.add_documents(chunks)
    print(f"✅ Embedded and stored {len(chunks)} chunks successfully")


def main():
    try:
        pdf_url = input("Enter PDF URL: ").strip()
        if not pdf_url:
            print("No URL provided. Exiting.")
            return

        start_time = time.time()

        pdf_bytes = download_pdf(pdf_url)
        chunks = split_pdf_chunks(pdf_bytes)
        index = init_pinecone()
        embed_and_store(chunks, index)

        total_time = time.time() - start_time
        print(f"🚀 Document ingestion completed in {total_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()