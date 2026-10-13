from rag_engine.retriever.retriever import Retriever


class SourceRetriever:

    def __init__(
        self,
        retriever=None,
        top_k=5
    ):

        self.retriever = retriever or Retriever(
            top_k=top_k
        )

    def retrieve(
        self,
        query,
        top_k=None
    ):

        results = self.retriever.retrieve(
            query,
            top_k
        )

        return results

    def retrieve_by_dataset(
        self,
        query,
        dataset,
        top_k=None
    ):

        results = self.retrieve(
            query,
            top_k
        )

        dataset = str(
            dataset
        ).strip().lower()

        filtered = []

        for result in results:

            result_dataset = str(
                result.get(
                    "dataset",
                    ""
                )
            ).strip().lower()

            if result_dataset == dataset:
                filtered.append(
                    result
                )

        return filtered

    def get_sources(
        self,
        results
    ):

        sources = []

        for result in results:

            source = {
                "rank": result.get(
                    "rank"
                ),
                "score": result.get(
                    "score"
                ),
                "dataset": result.get(
                    "dataset"
                ),
                "source": result.get(
                    "source"
                ),
                "document": result.get(
                    "document"
                )
            }

            sources.append(
                source
            )

        return sources

    def close(self):

        if hasattr(
            self.retriever,
            "client"
        ):

            try:

                self.retriever.client.close()

            except Exception:

                pass