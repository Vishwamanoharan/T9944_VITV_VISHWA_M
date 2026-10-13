
from collections import defaultdict
from typing import Callable, Dict, List

from src.events.event_schema import Event


class EventBus:
    """
    Simple in-process event bus.

    Agents subscribe to event types and publish events.
    """

    def __init__(self):

        self.subscribers: Dict[
            str,
            List[Callable]
        ] = defaultdict(list)

        self.event_history: List[Event] = []

    def subscribe(
        self,
        event_type: str,
        handler: Callable
    ):
        """
        Register an agent handler for an event type.
        """

        if handler not in self.subscribers[event_type]:

            self.subscribers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable
    ):
        """
        Remove an agent handler.
        """

        if event_type in self.subscribers:

            if handler in self.subscribers[event_type]:

                self.subscribers[event_type].remove(handler)

    def publish(self, event: Event):
        """
        Publish an event to all subscribed agents.
        """

        if not isinstance(event, Event):

            raise TypeError(
                "EventBus.publish() expects an Event object."
            )

        # Store event for memory/debugging/auditing.
        self.event_history.append(event)

        handlers = self.subscribers.get(
            event.event_type,
            []
        )

        for handler in handlers:

            handler(event)

    def get_history(self):
        """
        Return all published events.
        """

        return list(self.event_history)

    def get_subscriber_count(
        self,
        event_type: str
    ):
        """
        Return number of subscribers for an event type.
        """

        return len(
            self.subscribers.get(
                event_type,
                []
            )
        )

    def clear_history(self):
        """
        Clear event history.
        """

        self.event_history.clear()

