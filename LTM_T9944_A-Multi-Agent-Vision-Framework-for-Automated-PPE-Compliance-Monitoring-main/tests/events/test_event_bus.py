
from src.events.event_bus import EventBus
from src.events.event_schema import Event


def compliance_handler(event):

    print(
        f"Compliance Agent received: "
        f"{event.event_type}"
    )

    print(
        f"Worker: {event.worker_id}"
    )

    print(
        f"Payload: {event.payload}"
    )


def risk_handler(event):

    print(
        f"Risk Agent received: "
        f"{event.event_type}"
    )

    print(
        f"Worker: {event.worker_id}"
    )

    print(
        f"Payload: {event.payload}"
    )


def main():

    print("=" * 70)
    print("EVENT BUS TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Create Event Bus
    # --------------------------------------------------

    event_bus = EventBus()

    # --------------------------------------------------
    # Subscribe agents
    # --------------------------------------------------

    event_bus.subscribe(
        "PPE_VIOLATION_DETECTED",
        compliance_handler
    )

    event_bus.subscribe(
        "RISK_ASSESSED",
        risk_handler
    )

    # --------------------------------------------------
    # Create PPE violation event
    # --------------------------------------------------

    violation_event = Event(
        event_type="PPE_VIOLATION_DETECTED",
        source_agent="ComplianceAgent",
        worker_id="Worker-001",
        priority="HIGH",
        payload={
            "violation": "NO_HELMET",
            "severity": "HIGH"
        }
    )

    # --------------------------------------------------
    # Publish event
    # --------------------------------------------------

    print()
    print("Publishing PPE violation event...")

    event_bus.publish(
        violation_event
    )

    # --------------------------------------------------
    # Create Risk event
    # --------------------------------------------------

    risk_event = Event(
        event_type="RISK_ASSESSED",
        source_agent="RiskAgent",
        worker_id="Worker-001",
        priority="HIGH",
        payload={
            "risk_score": 40,
            "risk_level": "HIGH"
        }
    )

    print()
    print("Publishing risk event...")

    event_bus.publish(
        risk_event
    )

    # --------------------------------------------------
    # Validate history
    # --------------------------------------------------

    history = event_bus.get_history()

    print()
    print(
        f"Events stored in history: {len(history)}"
    )

    print(
        "PPE subscribers:",
        event_bus.get_subscriber_count(
            "PPE_VIOLATION_DETECTED"
        )
    )

    print(
        "Risk subscribers:",
        event_bus.get_subscriber_count(
            "RISK_ASSESSED"
        )
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert len(history) == 2

    assert history[0].event_type == (
        "PPE_VIOLATION_DETECTED"
    )

    assert history[1].event_type == (
        "RISK_ASSESSED"
    )

    assert (
        event_bus.get_subscriber_count(
            "PPE_VIOLATION_DETECTED"
        )
        == 1
    )

    assert (
        event_bus.get_subscriber_count(
            "RISK_ASSESSED"
        )
        == 1
    )

    print()
    print("=" * 70)
    print("EVENT BUS TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()

