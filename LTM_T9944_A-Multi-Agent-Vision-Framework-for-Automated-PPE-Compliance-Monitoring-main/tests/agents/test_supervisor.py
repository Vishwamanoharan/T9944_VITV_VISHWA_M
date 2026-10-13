from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes
from src.state.shared_state import SharedState

from src.agents.memory_agent.memory_agent import (
    MemoryAgent,
)

from src.agents.supervisor_agent.supervisor_agent import (
    SupervisorAgent,
)


def main():

    print("=" * 70)
    print("PHASE 7 SUPERVISOR AUTONOMOUS DECISION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Create system
    # ---------------------------------------------------------

    event_bus = EventBus()

    shared_state = SharedState()

    memory_agent = MemoryAgent(
        event_bus,
        shared_state,
    )

    supervisor = SupervisorAgent(
        event_bus,
        shared_state,
    )

    # ---------------------------------------------------------
    # Worker state
    # ---------------------------------------------------------

    worker = {
        "worker_id": "Worker-001",
        "helmet": False,
        "vest": True,
        "gloves": True,
        "boots": True,
        "goggles": True,
        "event": "NO_HELMET",
    }

    event_bus.publish(
        Event(
            event_type=(
                EventTypes.PPE_STATE_EVALUATED
            ),
            source_agent="ComplianceAgent",
            payload={
                "workers": [worker],
                "violations": [
                    {
                        "worker_id": "Worker-001",
                        "violation": "NO_HELMET",
                        "severity": "HIGH",
                    }
                ],
            },
        )
    )

    # ---------------------------------------------------------
    # Violation
    # ---------------------------------------------------------

    event_bus.publish(
        Event(
            event_type=(
                EventTypes.PPE_VIOLATION_DETECTED
            ),
            source_agent="ComplianceAgent",
            worker_id="Worker-001",
            priority="HIGH",
            payload={
                "worker_id": "Worker-001",
                "violation": "NO_HELMET",
                "severity": "HIGH",
            },
        )
    )

    # ---------------------------------------------------------
    # Risk
    # ---------------------------------------------------------

    event_bus.publish(
        Event(
            event_type=(
                EventTypes.RISK_ASSESSED
            ),
            source_agent="RiskAgent",
            worker_id="Worker-001",
            priority="HIGH",
            payload={
                "worker_id": "Worker-001",
                "risk_score": 40,
                "risk_level": "HIGH",
            },
        )
    )

    # ---------------------------------------------------------
    # Retrieve supervisor decision
    # ---------------------------------------------------------

    decision = shared_state.get_decision(
        "Worker-001"
    )

    # Supervisor should have generated a decision.
    assert decision is not None

    assert decision["worker_id"] == (
        "Worker-001"
    )

    assert decision["decision"] == (
        "SEND_WARNING"
    )

    # ---------------------------------------------------------
    # Verify analysis
    # ---------------------------------------------------------

    history = (
        shared_state.get_event_history(
            "Worker-001"
        )
    )

    analysis_events = [
        event
        for event in history
        if event.get("event_type")
        == EventTypes.SUPERVISOR_ANALYSIS
    ]

    assert len(analysis_events) == 1

    analysis = analysis_events[0]

    assert (
        analysis["payload"]["analysis"][
            "risk_level"
        ]
        == "HIGH"
    )

    # ---------------------------------------------------------
    # Verify RAG
    # ---------------------------------------------------------

    knowledge = analysis[
        "payload"
    ]["knowledge"]

    assert len(knowledge) > 0

    print()
    print(
        "Worker state retrieved successfully."
    )

    print(
        "Risk retrieved successfully."
    )

    print(
        "Supervisor analysis generated successfully."
    )

    print(
        "RAG knowledge retrieved successfully."
    )

    print(
        f"Decision: "
        f"{decision['decision']}"
    )

    print(
        f"Knowledge sources: "
        f"{decision['knowledge_sources']}"
    )

    print()
    print("=" * 70)
    print(
        "PHASE 7 SUPERVISOR TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()