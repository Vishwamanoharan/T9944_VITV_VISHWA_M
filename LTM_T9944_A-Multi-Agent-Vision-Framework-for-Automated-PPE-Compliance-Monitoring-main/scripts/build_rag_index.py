from pathlib import Path

from src.rag.document_loader import DocumentLoader
from src.rag.chunker import TextChunker
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


KNOWLEDGE_BASE = Path(
    "knowledge_base"
)

VECTOR_STORE_PATH = (
    "vector_db/ppe_rag"
)


def main():

    print("=" * 70)
    print("PHASE 6 - BUILD RAG KNOWLEDGE BASE")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load documents
    # ---------------------------------------------------------

    loader = DocumentLoader()

    documents = loader.load_directory(
        KNOWLEDGE_BASE
    )

    print()
    print(
        f"PDF pages loaded: "
        f"{len(documents)}"
    )

    if not documents:
        raise RuntimeError(
            "No PDF documents found in knowledge_base/"
        )

    # ---------------------------------------------------------
    # Chunk documents
    # ---------------------------------------------------------

    chunker = TextChunker(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = chunker.chunk_documents(
        documents
    )

    print(
        f"Text chunks created: "
        f"{len(chunks)}"
    )

    # ---------------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------------

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = (
        embedding_model.embed_documents(
            texts
        )
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # ---------------------------------------------------------
    # Build vector store
    # ---------------------------------------------------------

    vector_store = VectorStore(
        dimension=embeddings.shape[1],
        storage_path=VECTOR_STORE_PATH,
    )

    vector_store.add(
        embeddings,
        chunks,
    )

    vector_store.save()

    # ---------------------------------------------------------
    # Complete
    # ---------------------------------------------------------

    print()
    print(
        f"Vectors stored: "
        f"{len(chunks)}"
    )

    print(
        f"Vector database: "
        f"{VECTOR_STORE_PATH}"
    )

    print()
    print("=" * 70)
    print("RAG KNOWLEDGE BASE BUILD COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()