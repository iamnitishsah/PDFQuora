import os
import time
import requests
import tempfile
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "pdfquora"
NAMESPACE = "default"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

app = FastAPI(
    title="PDF Quora API",
    description="Send PDF URL and question together, get answer and auto-cleanup",
    version="2.0.0"
)


class QueryRequest(BaseModel):
    url: HttpUrl
    question: str


class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(INDEX_NAME)


def init_embedding():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )


def download_pdf(url: str) -> bytes:
    response = requests.get(str(url))
    response.raise_for_status()
    return response.content


def split_pdf_chunks(pdf_bytes: bytes) -> List:
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

    return chunks


def embed_and_store_chunks(chunks: List, index, embedding):
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embedding,
        namespace=NAMESPACE,
        text_key="page_content"
    )
    vector_store.add_documents(chunks)


def retrieve_context(query: str, embedding, index) -> List[Document]:
    query_vector = embedding.embed_query(query)

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        namespace=NAMESPACE,
        include_metadata=True
    )

    documents = []
    for match in results.matches:
        if match.metadata and "page_content" in match.metadata:
            documents.append(
                Document(
                    page_content=match.metadata["page_content"],
                    metadata={
                        **match.metadata,
                        "similarity_score": match.score
                    }
                )
            )

    return documents


def create_prompt(query: str, documents: List[Document]) -> str:
    if not documents:
        return f"No relevant information found for the query: {query}"

    context = "\n\n".join([
        f"Content {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(documents)
    ])

    prompt = f"""Based on the following document content, answer the user's question accurately and concisely.

Question: {query}

Document Content:
{context}

Answer: Provide a clear and detailed answer based only on the information given above. If the information is not sufficient, say so."""

    return prompt


def generate_answer(prompt: str) -> str:
    llm = GoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.0
    )

    response = llm.invoke(prompt)
    return response.strip()


def clear_index(index):
    index.delete(delete_all=True, namespace=NAMESPACE)


@app.get("/")
async def root():
    return {"message": "PDF Quora API - Send URL and question together!"}


@app.post("/query")
async def query_pdf(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        start_time = time.time()

        index = init_pinecone()
        print(f"Index created")

        pdf_bytes = download_pdf(request.url)
        print(f"PDF downloaded: {len(pdf_bytes)} bytes")

        chunks = split_pdf_chunks(pdf_bytes)
        print(f"Chunks split: {len(chunks)}")

        embedding = init_embedding()
        print(f"Embedding Model instance created: {embedding}")

        embed_and_store_chunks(chunks, index, embedding)
        print(f"Chunks embedded: {len(chunks)}")

        documents = retrieve_context(request.question, embedding, index)
        print(f"Documents retrieved: {len(documents)}")

        embedding_time = time.time() - start_time
        print(f"Total embedding time: {embedding_time}")

        if not documents:
            answer = "No relevant information found in the PDF for your question."
            confidence = "low"
        else:
            prompt = create_prompt(request.question, documents)
            answer = generate_answer(prompt)
            print(f"Answer generated: {answer}")
            confidence = "high" if len(documents) >= 3 else "medium"

        clear_index(index)

        response_time = time.time() - start_time

        return {
            "question": request.question,
            "answer": answer,
            "confidence": confidence,
            "chunks_processed": len(chunks),
            "response_time": round(response_time, 2),
            "status": "completed_and_cleaned"
        }

    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Failed to download PDF from URL")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)