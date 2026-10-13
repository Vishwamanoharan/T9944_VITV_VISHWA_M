from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Generates embeddings for RAG documents and queries.
    """

    def __init__(
        self,
        model_name="BAAI/bge-small-en-v1.5",
    ):

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def embed_documents(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings

    def embed_query(self, query):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        return embedding