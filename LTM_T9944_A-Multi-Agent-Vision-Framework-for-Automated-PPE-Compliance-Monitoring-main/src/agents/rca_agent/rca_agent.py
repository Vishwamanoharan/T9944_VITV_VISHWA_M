
from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes

from src.agents.rca_agent.rca_engine import RCAEngine


class RCAAgent:

    def __init__(self, event_bus: EventBus):

        self.event_bus = event_bus
        self.engine = RCAEngine()

        self.event_bus.subscribe(
            EventTypes.PPE_STATE_EVALUATED,
            self.handle_ppe_state,
        )

    def handle_ppe_state(self, event: Event):

        workers = event.payload["workers"]
        violations = event.payload["violations"]

        for worker in workers:

            result = self.engine.analyze(
                worker,
                violations,
            )

            self.event_bus.publish(
                Event(
                    event_type=EventTypes.RCA_COMPLETED,
                    source_agent="RCAAgent",
                    worker_id=worker["worker_id"],
                    payload=result,
                )
            )

