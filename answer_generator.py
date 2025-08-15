import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "pdfquora"
NAMESPACE = "pdf"
TOP_K = 5


class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def init_components():
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    return embedding, index


def retrieve_context(query, embedding, index):
    print(f"Searching for relevant content...")

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

    print(f"Found {len(documents)} relevant chunks")
    return documents


def create_prompt(query, documents):
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


def generate_answer(prompt):
    llm = GoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY
    )

    response = llm.invoke(prompt)
    return response.strip()


def main():
    print("=== PDF Quora - Ask questions about your PDF ===\n")

    try:
        embedding, index = init_components()

        while True:
            query = input("\nEnter your question (or 'quit' to exit): ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not query:
                print("Please enter a valid question.")
                continue

            start_time = time.time()

            documents = retrieve_context(query, embedding, index)

            if not documents:
                print("⚠️  No relevant information found in the PDF for your question.")
                continue

            prompt = create_prompt(query, documents)
            answer = generate_answer(prompt)

            print(f"\n📝 Answer:")
            print(f"{answer}")

            response_time = time.time() - start_time
            print(f"\n⏱️  Response time: {response_time:.2f} seconds")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()