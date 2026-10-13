from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


class RAGRetriever:
    """
    Retrieves relevant safety knowledge
    from the FAISS knowledge base.
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
    ):

        self.vector_store = vector_store

        self.embedding_model = embedding_model

    def retrieve(
        self,
        query,
        top_k=5,
    ):

        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        return results