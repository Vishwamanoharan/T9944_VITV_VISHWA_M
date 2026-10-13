from collections import defaultdict

from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes
from src.state.shared_state import SharedState


class MemoryAgent:
    """
    Memory Agent maintains:

    1. Per-worker historical event memory
    2. Central SharedState containing the latest system state

    Phase 7 additions:
    - Stores supervisor analysis events
    - Stores supervisor decisions
    """

    def __init__(
        self,
        event_bus: EventBus,
        shared_state: SharedState,
    ):
        self.event_bus = event_bus
        self.shared_state = shared_state

        self.worker_history = defaultdict(list)

        # PPE state
        self.event_bus.subscribe(
            EventTypes.PPE_STATE_EVALUATED,
            self.handle_ppe_state,
        )

        # PPE violations
        self.event_bus.subscribe(
            EventTypes.PPE_VIOLATION_DETECTED,
            self.handle_violation,
        )

        # Risk
        self.event_bus.subscribe(
            EventTypes.RISK_ASSESSED,
            self.handle_risk,
        )

        # RCA
        self.event_bus.subscribe(
            EventTypes.RCA_COMPLETED,
            self.handle_rca,
        )

        # Recommendation
        self.event_bus.subscribe(
            EventTypes.RECOMMENDATION_CREATED,
            self.handle_recommendation,
        )

        # Alert
        self.event_bus.subscribe(
            EventTypes.ALERT_GENERATED,
            self.handle_alert,
        )

        # Supervisor analysis
        self.event_bus.subscribe(
            EventTypes.SUPERVISOR_ANALYSIS,
            self.handle_supervisor_analysis,
        )

        # Supervisor decision
        self.event_bus.subscribe(
            EventTypes.SUPERVISOR_DECISION,
            self.handle_decision,
        )

    # ============================================================
    # GENERIC EVENT STORAGE
    # ============================================================

    def _store(self, event: Event):
        """
        Store an event in the worker's historical memory.
        """

        worker_id = event.worker_id

        if worker_id is None:
            return

        record = {
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "payload": event.payload,
        }

        self.worker_history[worker_id].append(
            record
        )

    # ============================================================
    # PPE STATE
    # ============================================================

    def handle_ppe_state(self, event: Event):
        """
        Store the latest worker PPE state.

        Also records the PPE state event in SharedState.
        """

        workers = event.payload["workers"]

        violations = event.payload.get(
            "violations",
            [],
        )

        for worker in workers:

            worker_id = worker["worker_id"]

            # Update current worker state
            self.shared_state.update_worker(
                worker
            )

            # ----------------------------------------------------
            # Maintain CURRENT violations for this worker
            # ----------------------------------------------------

            worker_violations = [
                violation
                for violation in violations
                if violation.get("worker_id")
                == worker_id
            ]

            self.shared_state.update_violations(
                worker_id,
                worker_violations,
            )

            # ----------------------------------------------------
            # Store worker PPE state in historical memory
            # ----------------------------------------------------

            record = {
                "event_type": (
                    EventTypes.PPE_STATE_EVALUATED
                ),
                "timestamp": event.timestamp,
                "payload": worker,
            }

            self.worker_history[
                worker_id
            ].append(record)

        # Store complete event
        self.shared_state.record_event(
            event
        )

    # ============================================================
    # PPE VIOLATION
    # ============================================================

    def handle_violation(self, event: Event):
        """
        Store a PPE violation event.
        """

        worker_id = event.worker_id

        if worker_id is None:
            return

        self._store(event)

        existing = (
            self.shared_state.get_violations(
                worker_id
            )
        )

        violation = event.payload

        if violation not in existing:
            existing.append(
                violation
            )

        self.shared_state.update_violations(
            worker_id,
            existing,
        )

        self.shared_state.record_event(
            event
        )

    # ============================================================
    # RISK
    # ============================================================

    def handle_risk(self, event: Event):
        """
        Store current risk assessment.
        """

        worker_id = event.worker_id

        if worker_id is None:
            return

        self._store(event)

        self.shared_state.update_risk(
            worker_id,
            event.payload,
        )

        self.shared_state.record_event(
            event
        )

    # ============================================================
    # RCA
    # ============================================================

    def handle_rca(self, event: Event):
        """
        Store RCA result.
        """

        worker_id = event.worker_id

        if worker_id is None:
            return

        self._store(event)

        self.shared_state.update_rca(
            worker_id,
            event.payload,
        )

        self.shared_state.record_event(
            event
        )

    # ============================================================
    # RECOMMENDATION
    # ============================================================

    def handle_recommendation(
        self,
        event: Event,
    ):
        """
        Store recommendation generated
        by RecommendationAgent.
        """

        worker_id = event.worker_id

        if worker_id is None:
            return

        self._store(event)

        self.shared_state.update_recommendation(
            worker_id,
            event.payload,
        )

        self.shared_state.record_event(
            event
        )

    # ============================================================
    # ALERT
    # ============================================================

    def handle_alert(self, event: Event):
        """
        Store generated alert.
        """

        worker_id = event.worker_id

        if worker_id is None:
            return

        self._store(event)

        self.shared_state.update_alert(
            worker_id,
            event.payload,
        )

        self.shared_state.record_event(
            event
        )

    # ============================================================
    # SUPERVISOR ANALYSIS - PHASE 7
    # ============================================================

    def handle_supervisor_analysis(
        self,
        event: Event,
    ):
        """
        Store supervisor analysis.

        The analysis contains:
        - Risk information
        - Violation information
        - Historical information
        - Retrieved RAG knowledge
        """

        worker_id = event.worker_id

        if worker_id is None:
            return

        # Store in worker historical memory
        self._store(event)

        # Store complete event in SharedState
        self.shared_state.record_event(
            event
        )

    # ============================================================
    # SUPERVISOR DECISION - PHASE 7
    # ============================================================

    def handle_decision(
        self,
        event: Event,
    ):
        """
        Store final supervisor decision.
        """

        worker_id = event.worker_id

        if worker_id is None:
            return

        self._store(event)

        self.shared_state.update_decision(
            worker_id,
            event.payload,
        )

        self.shared_state.record_event(
            event
        )

    # ============================================================
    # MEMORY ACCESS
    # ============================================================

    def get_history(self, worker_id):
        """
        Return historical events for a worker.
        """

        return list(
            self.worker_history.get(
                worker_id,
                [],
            )
        )

    def get_worker_state(
        self,
        worker_id,
    ):
        """
        Return current worker state.
        """

        return self.shared_state.get_worker(
            worker_id
        )

    def get_worker_snapshot(
        self,
        worker_id,
    ):
        """
        Return complete current worker
        state and historical information.
        """

        return self.shared_state.get_worker_snapshot(
            worker_id
        )

    def get_all_state(self):
        """
        Return complete shared system state.
        """

        return self.shared_state.snapshot()