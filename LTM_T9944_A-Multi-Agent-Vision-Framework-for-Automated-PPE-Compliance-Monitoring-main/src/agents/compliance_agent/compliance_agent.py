
from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes


class ComplianceAgent:

    def __init__(self, event_bus: EventBus):

        self.event_bus = event_bus

    def evaluate(self, workers):

        violations = []

        worker_states = []

        for worker in workers:

            worker_id = worker["worker_id"]

            worker_state = {
                "worker_id": worker_id,
                "helmet": worker.get("helmet", True),
                "vest": worker.get("vest", True),
                "gloves": worker.get("gloves", True),
                "boots": worker.get("boots", True),
                "goggles": worker.get("goggles", True),
                "event": worker.get("event", "NORMAL"),
            }

            worker_states.append(worker_state)

            if not worker_state["helmet"]:
                violations.append({
                    "worker_id": worker_id,
                    "violation": "NO_HELMET",
                    "severity": "HIGH",
                })

            if not worker_state["vest"]:
                violations.append({
                    "worker_id": worker_id,
                    "violation": "NO_VEST",
                    "severity": "HIGH",
                })

            if not worker_state["gloves"]:
                violations.append({
                    "worker_id": worker_id,
                    "violation": "NO_GLOVES",
                    "severity": "MEDIUM",
                })

            if not worker_state["boots"]:
                violations.append({
                    "worker_id": worker_id,
                    "violation": "NO_BOOTS",
                    "severity": "MEDIUM",
                })

            if not worker_state["goggles"]:
                violations.append({
                    "worker_id": worker_id,
                    "violation": "NO_GOGGLES",
                    "severity": "MEDIUM",
                })

        # Publish evaluated PPE state
        self.event_bus.publish(
            Event(
                event_type=EventTypes.PPE_STATE_EVALUATED,
                source_agent="ComplianceAgent",
                payload={
                    "workers": worker_states,
                    "violations": violations,
                },
            )
        )

        # Publish individual violation events
        for violation in violations:

            priority = violation["severity"]

            self.event_bus.publish(
                Event(
                    event_type=EventTypes.PPE_VIOLATION_DETECTED,
                    source_agent="ComplianceAgent",
                    worker_id=violation["worker_id"],
                    priority=priority,
                    payload=violation,
                )
            )

        return violations

