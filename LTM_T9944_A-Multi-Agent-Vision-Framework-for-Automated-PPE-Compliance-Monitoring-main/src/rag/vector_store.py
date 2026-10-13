from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """
    FAISS vector store for the PPE knowledge base.
    """

    def __init__(
        self,
        dimension=384,
        storage_path="vector_db/ppe_rag",
    ):

        self.dimension = dimension

        self.storage_path = Path(
            storage_path
        )

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.documents = []

    # ---------------------------------------------------------
    # Add documents
    # ---------------------------------------------------------

    def add(
        self,
        embeddings,
        documents,
    ):

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.index.add(embeddings)

        self.documents.extend(documents)

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query_embedding,
        top_k=5,
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_embedding,
            min(
                top_k,
                len(self.documents),
            ),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            document = dict(
                self.documents[index]
            )

            document["score"] = float(
                score
            )

            results.append(document)

        return results

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(self):

        index_path = (
            self.storage_path
            / "index.faiss"
        )

        metadata_path = (
            self.storage_path
            / "documents.npy"
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        np.save(
            metadata_path,
            np.array(
                self.documents,
                dtype=object,
            ),
            allow_pickle=True,
        )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def load(self):

        index_path = (
            self.storage_path
            / "index.faiss"
        )

        metadata_path = (
            self.storage_path
            / "documents.npy"
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: "
                f"{metadata_path}"
            )

        self.index = faiss.read_index(
            str(index_path)
        )

        self.documents = np.load(
            metadata_path,
            allow_pickle=True,
        ).tolist()