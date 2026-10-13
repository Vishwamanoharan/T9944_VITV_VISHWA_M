from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import RAGRetriever


VECTOR_STORE_PATH = (
    "vector_db/ppe_rag"
)


def main():

    print("=" * 70)
    print("PHASE 6 RAG RETRIEVAL TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    embedding_model = EmbeddingModel()

    # ---------------------------------------------------------
    # Load vector store
    # ---------------------------------------------------------

    vector_store = VectorStore(
        dimension=384,
        storage_path=VECTOR_STORE_PATH,
    )

    vector_store.load()

    print()
    print(
        f"Knowledge chunks: "
        f"{len(vector_store.documents)}"
    )

    # ---------------------------------------------------------
    # Create retriever
    # ---------------------------------------------------------

    retriever = RAGRetriever(
        vector_store,
        embedding_model,
    )

    # ---------------------------------------------------------
    # Test query
    # ---------------------------------------------------------

    query = (
        "What personal protective equipment "
        "is required for construction workers?"
    )

    results = retriever.retrieve(
        query,
        top_k=3,
    )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    assert len(results) > 0

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"RESULT {index}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Category: "
            f"{result['category']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        preview = (
            result["text"]
            .replace("\n", " ")
        )

        print(
            f"Text: "
            f"{preview[:300]}..."
        )

    print()
    print("=" * 70)
    print("PHASE 6 RAG RETRIEVAL TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()