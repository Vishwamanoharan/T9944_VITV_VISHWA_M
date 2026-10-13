from rag_engine.retriever.source_retriever import (
    SourceRetriever
)

from rag_engine.reranker.reranker import (
    Reranker
)

from rag_engine.retriever.context_builder import (
    RAGContextBuilder
)

from llm.prompt_manager.telecom_qa_prompt import (
    TelecomQAPrompt
)

from llm.qwen_2_5.inference import (
    QwenInference
)


class ResponseGenerator:

    def __init__(self):

        print(
            "\nInitializing RAG Response Generator..."
        )

        self.retriever = (
            SourceRetriever()
        )

        self.reranker = (
            Reranker()
        )

        self.context_builder = (
            RAGContextBuilder()
        )

        self.prompt_manager = (
            TelecomQAPrompt()
        )

        self.llm = (
            QwenInference()
        )

        print(
            "RAG Response Generator initialized."
        )

    def generate(
        self,
        query,
        retrieval_top_k=5,
        rerank_top_k=3,
        max_new_tokens=256
    ):

        if query is None:

            raise ValueError(
                "Query cannot be None."
            )

        query = str(
            query
        ).strip()

        if not query:

            raise ValueError(
                "Query cannot be empty."
            )

        print("\n" + "=" * 60)
        print("RAG QUERY")
        print("=" * 60)

        print(
            f"\nQuery:\n{query}"
        )

        # --------------------------------------------------
        # STEP 1: RETRIEVAL
        # --------------------------------------------------

        print(
            "\nRetrieving documents..."
        )

        retrieved_documents = (
            self.retriever.retrieve(
                query,
                top_k=retrieval_top_k
            )
        )

        print(
            f"Retrieved documents: "
            f"{len(retrieved_documents)}"
        )

        # --------------------------------------------------
        # STEP 2: RERANKING
        # --------------------------------------------------

        print(
            "\nReranking documents..."
        )

        reranked_documents = (
            self.reranker.rerank(
                query,
                retrieved_documents,
                top_k=rerank_top_k
            )
        )

        print(
            f"Reranked documents: "
            f"{len(reranked_documents)}"
        )

        # --------------------------------------------------
        # STEP 3: CONTEXT BUILDING
        # --------------------------------------------------

        print(
            "\nBuilding RAG context..."
        )

        context_result = (
            self.context_builder.build(
                query=query,
                documents=reranked_documents
            )
        )

        context = (
            context_result["context"]
        )

        sources = (
            context_result["sources"]
        )

        print(
            f"Context length: "
            f"{len(context)} characters"
        )

        print(
            f"Sources included: "
            f"{len(sources)}"
        )

        # --------------------------------------------------
        # STEP 4: PROMPT BUILDING
        # --------------------------------------------------

        print(
            "\nBuilding LLM prompt..."
        )

        prompt = (
            self.prompt_manager.build(
                query=query,
                context=context
            )
        )

        # --------------------------------------------------
        # STEP 5: LLM INFERENCE
        # --------------------------------------------------

        print(
            "\nGenerating answer with Qwen..."
        )

        answer = (
            self.llm.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
        )

        # --------------------------------------------------
        # FINAL OUTPUT
        # --------------------------------------------------

        return {

            "query":
                query,

            "answer":
                answer,

            "sources":
                sources,

            "retrieved_documents":
                retrieved_documents,

            "reranked_documents":
                reranked_documents,

            "context":
                context
        }

    def close(self):

        try:

            self.retriever.close()

        except Exception:

            pass

        print(
            "RAG Response Generator closed."
        )