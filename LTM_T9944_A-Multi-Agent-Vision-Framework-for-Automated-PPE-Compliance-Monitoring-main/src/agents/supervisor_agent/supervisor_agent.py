from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes

from src.state.shared_state import SharedState

from src.agents.supervisor_agent.decision_engine import (
    SupervisorDecisionEngine,
)

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import RAGRetriever


class SupervisorAgent:

    def __init__(
        self,
        event_bus: EventBus,
        shared_state: SharedState,
        vector_store_path="vector_db/ppe_rag",
    ):

        self.event_bus = event_bus

        self.shared_state = shared_state

        self.decision_engine = (
            SupervisorDecisionEngine()
        )

        # -----------------------------------------------------
        # RAG components
        # -----------------------------------------------------

        self.embedding_model = (
            EmbeddingModel()
        )

        self.vector_store = VectorStore(
            dimension=384,
            storage_path=vector_store_path,
        )

        self.vector_store.load()

        self.retriever = RAGRetriever(
            self.vector_store,
            self.embedding_model,
        )

        # -----------------------------------------------------
        # Subscribe to risk events
        # -----------------------------------------------------

        self.event_bus.subscribe(
            EventTypes.RISK_ASSESSED,
            self.handle_risk,
        )

    # =========================================================
    # Risk event
    # =========================================================

    def handle_risk(self, event: Event):

        worker_id = event.worker_id

        if worker_id is None:
            return

        risk = event.payload

        # -----------------------------------------------------
        # Current worker information
        # -----------------------------------------------------

        worker = (
            self.shared_state.get_worker(
                worker_id
            )
        )

        violations = (
            self.shared_state.get_violations(
                worker_id
            )
        )

        history = (
            self.shared_state.get_event_history(
                worker_id
            )
        )

        # -----------------------------------------------------
        # Build RAG query
        # -----------------------------------------------------

        violation_names = [
            violation.get(
                "violation",
                "",
            )
            for violation in violations
        ]

        query = (
            "Construction PPE safety requirements "
            "and recommended actions for worker "
            f"{worker_id}. "
            f"Risk level: {risk.get('risk_level')}. "
            f"Risk score: {risk.get('risk_score')}. "
            f"Current PPE violations: "
            f"{', '.join(violation_names)}."
        )

        # -----------------------------------------------------
        # Retrieve knowledge
        # -----------------------------------------------------

        retrieved_knowledge = (
            self.retriever.retrieve(
                query,
                top_k=3,
            )
        )

        # -----------------------------------------------------
        # Supervisor analysis
        # -----------------------------------------------------

        analysis = (
            self.decision_engine.decide(
                risk=risk,
                violations=violations,
                history=history,
                retrieved_knowledge=(
                    retrieved_knowledge
                ),
            )
        )

        # -----------------------------------------------------
        # Publish analysis
        # -----------------------------------------------------

        self.event_bus.publish(
            Event(
                event_type=(
                    EventTypes.SUPERVISOR_ANALYSIS
                ),
                source_agent="SupervisorAgent",
                worker_id=worker_id,
                priority=analysis["priority"],
                payload={
                    "worker_id": worker_id,
                    "analysis": analysis,
                    "knowledge": [
                        {
                            "source": item.get(
                                "source"
                            ),
                            "page": item.get(
                                "page"
                            ),
                            "category": item.get(
                                "category"
                            ),
                            "score": item.get(
                                "score"
                            ),
                            "text": item.get(
                                "text"
                            ),
                        }
                        for item in retrieved_knowledge
                    ],
                },
            )
        )

        # -----------------------------------------------------
        # Publish final decision
        # -----------------------------------------------------

        result = {
            "worker_id": worker_id,
            "decision": analysis["decision"],
            "risk_level": analysis["risk_level"],
            "risk_score": analysis["risk_score"],
            "historical_violations": (
                analysis[
                    "historical_violations"
                ]
            ),
            "knowledge_sources": (
                analysis[
                    "knowledge_sources"
                ]
            ),
        }

        self.event_bus.publish(
            Event(
                event_type=(
                    EventTypes.SUPERVISOR_DECISION
                ),
                source_agent="SupervisorAgent",
                worker_id=worker_id,
                priority=analysis["priority"],
                payload=result,
            )
        )