
from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes


class AlertAgent:

    def __init__(self, event_bus: EventBus):

        self.event_bus = event_bus

        self.event_bus.subscribe(
            EventTypes.RECOMMENDATION_CREATED,
            self.handle_recommendation,
        )

    def handle_recommendation(self, event: Event):

        recommendation = event.payload

        if recommendation["priority"] not in [
            "HIGH",
            "CRITICAL",
        ]:
            return

        alert = {
            "worker_id": recommendation["worker_id"],
            "action": recommendation["action"],
            "priority": recommendation["priority"],
        }

        self.event_bus.publish(
            Event(
                event_type=EventTypes.ALERT_GENERATED,
                source_agent="AlertAgent",
                worker_id=recommendation["worker_id"],
                priority=recommendation["priority"],
                payload=alert,
            )
        )

