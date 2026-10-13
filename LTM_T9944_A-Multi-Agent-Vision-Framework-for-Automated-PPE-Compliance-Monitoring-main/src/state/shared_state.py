from copy import deepcopy
from datetime import datetime, timezone


class SharedState:
    """
    Central shared state for the multi-agent PPE monitoring system.

    Stores:
    - Current worker state
    - Current PPE violations
    - Current risk assessment
    - Current RCA result
    - Current recommendation
    - Current alert
    - Current supervisor decision
    - Complete event history
    """

    def __init__(self):
        # Current state
        self.workers = {}

        self.violations = {}
        self.risks = {}
        self.rca_results = {}
        self.recommendations = {}
        self.alerts = {}
        self.decisions = {}

        # Historical events
        self.event_history = []

        # Last state update timestamp
        self.last_updated = None

    # =========================================================
    # INTERNAL
    # =========================================================

    def _touch(self):
        """Update the last modified timestamp."""
        self.last_updated = datetime.now(
            timezone.utc
        ).isoformat()

    # =========================================================
    # WORKER STATE
    # =========================================================

    def update_worker(self, worker):
        """Store the latest state of one worker."""

        worker_id = worker["worker_id"]

        self.workers[worker_id] = deepcopy(worker)

        self._touch()

    def update_workers(self, workers):
        """Store the latest state of multiple workers."""

        for worker in workers:
            self.update_worker(worker)

    def get_worker(self, worker_id):
        """Return the current state of one worker."""

        worker = self.workers.get(worker_id)

        if worker is None:
            return None

        return deepcopy(worker)

    def get_all_workers(self):
        """Return the current state of all workers."""

        return deepcopy(self.workers)

    # =========================================================
    # PPE VIOLATIONS
    # =========================================================

    def update_violations(self, worker_id, violations):
        """Store the latest violations for a worker."""

        self.violations[worker_id] = deepcopy(
            violations
        )

        self._touch()

    def get_violations(self, worker_id):
        """Return the current violations for a worker."""

        return deepcopy(
            self.violations.get(worker_id, [])
        )

    # =========================================================
    # RISK
    # =========================================================

    def update_risk(self, worker_id, risk):
        """Store the latest risk assessment."""

        self.risks[worker_id] = deepcopy(risk)

        self._touch()

    def get_risk(self, worker_id):
        """Return the latest risk assessment."""

        return deepcopy(
            self.risks.get(worker_id)
        )

    # =========================================================
    # ROOT CAUSE ANALYSIS
    # =========================================================

    def update_rca(self, worker_id, rca):
        """Store the latest RCA result."""

        self.rca_results[worker_id] = deepcopy(rca)

        self._touch()

    def get_rca(self, worker_id):
        """Return the latest RCA result."""

        return deepcopy(
            self.rca_results.get(worker_id)
        )

    # =========================================================
    # RECOMMENDATION
    # =========================================================

    def update_recommendation(
        self,
        worker_id,
        recommendation,
    ):
        """Store the latest recommendation."""

        self.recommendations[worker_id] = deepcopy(
            recommendation
        )

        self._touch()

    def get_recommendation(self, worker_id):
        """Return the latest recommendation."""

        return deepcopy(
            self.recommendations.get(worker_id)
        )

    # =========================================================
    # ALERT
    # =========================================================

    def update_alert(self, worker_id, alert):
        """Store the latest alert."""

        self.alerts[worker_id] = deepcopy(alert)

        self._touch()

    def get_alert(self, worker_id):
        """Return the latest alert."""

        return deepcopy(
            self.alerts.get(worker_id)
        )

    # =========================================================
    # SUPERVISOR DECISION
    # =========================================================

    def update_decision(self, worker_id, decision):
        """Store the latest supervisor decision."""

        self.decisions[worker_id] = deepcopy(decision)

        self._touch()

    def get_decision(self, worker_id):
        """Return the latest supervisor decision."""

        return deepcopy(
            self.decisions.get(worker_id)
        )

    # =========================================================
    # EVENT HISTORY
    # =========================================================

    def record_event(self, event):
        """
        Store an event in the global event history.
        """

        self.event_history.append(
            deepcopy(event.to_dict())
        )

        self._touch()

    def get_event_history(self, worker_id=None):
        """
        Return event history.

        If worker_id is None:
            Return all events.

        If worker_id is provided:
            Return events directly associated with that worker,
            including multi-worker events such as
            PPE_STATE_EVALUATED.
        """

        # Return complete history
        if worker_id is None:
            return deepcopy(
                self.event_history
            )

        worker_events = []

        for event in self.event_history:

            # -------------------------------------------------
            # Direct worker-specific event
            # -------------------------------------------------

            if event.get("worker_id") == worker_id:

                worker_events.append(
                    deepcopy(event)
                )

                continue

            # -------------------------------------------------
            # Multi-worker event
            # -------------------------------------------------

            payload = event.get(
                "payload",
                {}
            )

            workers = payload.get(
                "workers",
                []
            )

            for worker in workers:

                if worker.get(
                    "worker_id"
                ) == worker_id:

                    worker_events.append(
                        deepcopy(event)
                    )

                    break

        return worker_events

    # =========================================================
    # COMPLETE WORKER SNAPSHOT
    # =========================================================

    def get_worker_snapshot(self, worker_id):
        """
        Return the complete current and historical
        information for a worker.
        """

        return {
            "worker": self.get_worker(
                worker_id
            ),

            "violations": self.get_violations(
                worker_id
            ),

            "risk": self.get_risk(
                worker_id
            ),

            "rca": self.get_rca(
                worker_id
            ),

            "recommendation": self.get_recommendation(
                worker_id
            ),

            "alert": self.get_alert(
                worker_id
            ),

            "decision": self.get_decision(
                worker_id
            ),

            "history": self.get_event_history(
                worker_id
            ),
        }

    # =========================================================
    # COMPLETE SYSTEM SNAPSHOT
    # =========================================================

    def snapshot(self):
        """
        Return the complete shared system state.
        """

        return {
            "workers": deepcopy(
                self.workers
            ),

            "violations": deepcopy(
                self.violations
            ),

            "risks": deepcopy(
                self.risks
            ),

            "rca_results": deepcopy(
                self.rca_results
            ),

            "recommendations": deepcopy(
                self.recommendations
            ),

            "alerts": deepcopy(
                self.alerts
            ),

            "decisions": deepcopy(
                self.decisions
            ),

            "event_history": deepcopy(
                self.event_history
            ),

            "last_updated": self.last_updated,
        }

    # =========================================================
    # CLEAR STATE
    # =========================================================

    def clear(self):
        """
        Clear all current state and historical events.
        """

        self.workers.clear()

        self.violations.clear()

        self.risks.clear()

        self.rca_results.clear()

        self.recommendations.clear()

        self.alerts.clear()

        self.decisions.clear()

        self.event_history.clear()

        self.last_updated = None