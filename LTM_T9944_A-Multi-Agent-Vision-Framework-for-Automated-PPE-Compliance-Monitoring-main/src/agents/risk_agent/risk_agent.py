
from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes

from src.agents.risk_agent.risk_model import RiskModel


class RiskAgent:

    def __init__(self, event_bus: EventBus):

        self.event_bus = event_bus
        self.model = RiskModel()

        self.event_bus.subscribe(
            EventTypes.PPE_STATE_EVALUATED,
            self.handle_ppe_state,
        )

    def handle_ppe_state(self, event: Event):

        workers = event.payload["workers"]

        risk_results = []

        for worker in workers:

            risk = self.model.calculate(worker)

            result = {
                "worker_id": worker["worker_id"],
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
            }

            risk_results.append(result)

            priority = risk["risk_level"]

            self.event_bus.publish(
                Event(
                    event_type=EventTypes.RISK_ASSESSED,
                    source_agent="RiskAgent",
                    worker_id=worker["worker_id"],
                    priority=priority,
                    payload=result,
                )
            )

        return risk_results
