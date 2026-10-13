
from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes

from src.agents.recommendation_agent.action_generator import (
    ActionGenerator
)


class RecommendationAgent:

    def __init__(self, event_bus: EventBus):

        self.event_bus = event_bus
        self.generator = ActionGenerator()

        self.event_bus.subscribe(
            EventTypes.RISK_ASSESSED,
            self.handle_risk,
        )

    def handle_risk(self, event: Event):

        risk = event.payload

        action = self.generator.generate(risk)

        recommendation = {
            "worker_id": risk["worker_id"],
            "action": action["action"],
            "priority": action["priority"],
        }

        self.event_bus.publish(
            Event(
                event_type=EventTypes.RECOMMENDATION_CREATED,
                source_agent="RecommendationAgent",
                worker_id=risk["worker_id"],
                priority=action["priority"],
                payload=recommendation,
            )
        )

