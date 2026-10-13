from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes
from src.state.shared_state import SharedState
from src.agents.memory_agent.memory_agent import MemoryAgent


def main():

    print("=" * 70)
    print("PHASE 5 SHARED STATE + MEMORY TEST")
    print("=" * 70)

    event_bus = EventBus()
    shared_state = SharedState()

    memory_agent = MemoryAgent(
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
            event_type=EventTypes.PPE_STATE_EVALUATED,
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
            event_type=EventTypes.PPE_VIOLATION_DETECTED,
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
            event_type=EventTypes.RISK_ASSESSED,
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
    # RCA
    # ---------------------------------------------------------

    event_bus.publish(
        Event(
            event_type=EventTypes.RCA_COMPLETED,
            source_agent="RCAAgent",
            worker_id="Worker-001",
            payload={
                "worker_id": "Worker-001",
                "root_cause": "HELMET_NON_COMPLIANCE",
                "confidence": 0.85,
            },
        )
    )

    # ---------------------------------------------------------
    # Recommendation
    # ---------------------------------------------------------

    event_bus.publish(
        Event(
            event_type=EventTypes.RECOMMENDATION_CREATED,
            source_agent="RecommendationAgent",
            worker_id="Worker-001",
            priority="HIGH",
            payload={
                "worker_id": "Worker-001",
                "action": "ISSUE_IMMEDIATE_WARNING",
                "priority": "HIGH",
            },
        )
    )

    # ---------------------------------------------------------
    # Alert
    # ---------------------------------------------------------

    event_bus.publish(
        Event(
            event_type=EventTypes.ALERT_GENERATED,
            source_agent="AlertAgent",
            worker_id="Worker-001",
            priority="HIGH",
            payload={
                "worker_id": "Worker-001",
                "action": "ISSUE_IMMEDIATE_WARNING",
                "priority": "HIGH",
            },
        )
    )

    # ---------------------------------------------------------
    # Supervisor decision
    # ---------------------------------------------------------

    event_bus.publish(
        Event(
            event_type=EventTypes.SUPERVISOR_DECISION,
            source_agent="SupervisorAgent",
            worker_id="Worker-001",
            priority="HIGH",
            payload={
                "worker_id": "Worker-001",
                "decision": "SEND_WARNING",
                "risk_level": "HIGH",
            },
        )
    )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    worker_state = shared_state.get_worker(
        "Worker-001"
    )

    assert worker_state is not None
    assert worker_state["helmet"] is False

    violations = shared_state.get_violations(
        "Worker-001"
    )

    assert len(violations) == 1
    assert violations[0]["violation"] == "NO_HELMET"

    risk = shared_state.get_risk(
        "Worker-001"
    )

    assert risk["risk_level"] == "HIGH"
    assert risk["risk_score"] == 40

    rca = shared_state.get_rca(
        "Worker-001"
    )

    assert rca["root_cause"] == "HELMET_NON_COMPLIANCE"

    recommendation = shared_state.get_recommendation(
        "Worker-001"
    )

    assert recommendation["action"] == "ISSUE_IMMEDIATE_WARNING"

    alert = shared_state.get_alert(
        "Worker-001"
    )

    assert alert["priority"] == "HIGH"

    decision = shared_state.get_decision(
        "Worker-001"
    )

    assert decision["decision"] == "SEND_WARNING"

    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------

    history = shared_state.get_event_history(
        "Worker-001"
    )

    assert len(history) == 7

    # Memory Agent history
    memory_history = memory_agent.get_history(
        "Worker-001"
    )

    assert len(memory_history) == 7

    # Complete worker snapshot
    snapshot = memory_agent.get_worker_snapshot(
        "Worker-001"
    )

    assert snapshot["worker"] is not None
    assert snapshot["risk"] is not None
    assert snapshot["rca"] is not None
    assert snapshot["recommendation"] is not None
    assert snapshot["alert"] is not None
    assert snapshot["decision"] is not None
    assert len(snapshot["history"]) == 7

    print()
    print("Worker state stored successfully.")
    print("Violation state stored successfully.")
    print("Risk state stored successfully.")
    print("RCA state stored successfully.")
    print("Recommendation state stored successfully.")
    print("Alert state stored successfully.")
    print("Supervisor decision stored successfully.")
    print("Event history stored successfully.")
    print("Worker snapshot retrieved successfully.")

    print()
    print("=" * 70)
    print("PHASE 5 SHARED STATE + MEMORY TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()