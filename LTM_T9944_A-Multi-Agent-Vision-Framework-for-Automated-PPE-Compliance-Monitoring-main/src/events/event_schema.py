
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict
import uuid


@dataclass
class Event:
    """
    Standard event exchanged between agents.
    """

    event_type: str
    source_agent: str
    payload: Dict[str, Any]

    worker_id: str | None = None
    priority: str = "NORMAL"

    event_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    timestamp: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def to_dict(self):
        """
        Convert event into a dictionary.
        """

        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_agent": self.source_agent,
            "worker_id": self.worker_id,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Reconstruct an Event from a dictionary.
        """

        return cls(
            event_type=data["event_type"],
            source_agent=data["source_agent"],
            payload=data.get("payload", {}),
            worker_id=data.get("worker_id"),
            priority=data.get("priority", "NORMAL"),
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=data.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat()
            ),
        )

