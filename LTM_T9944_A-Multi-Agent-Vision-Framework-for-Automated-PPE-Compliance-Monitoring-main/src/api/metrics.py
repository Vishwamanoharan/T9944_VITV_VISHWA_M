from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# ============================================================
# SYSTEM METRICS
# ============================================================

TOTAL_EVENTS = Counter(
    "ppe_total_events",
    "Total number of events processed",
)

TOTAL_WORKERS = Gauge(
    "ppe_total_workers",
    "Total number of workers currently tracked",
)

TOTAL_VIOLATIONS = Gauge(
    "ppe_total_current_violations",
    "Total current PPE violations",
)

HIGH_RISK_WORKERS = Gauge(
    "ppe_high_risk_workers",
    "Number of workers with HIGH risk",
)

CRITICAL_RISK_WORKERS = Gauge(
    "ppe_critical_risk_workers",
    "Number of workers with CRITICAL risk",
)

MEDIUM_RISK_WORKERS = Gauge(
    "ppe_medium_risk_workers",
    "Number of workers with MEDIUM risk",
)

LOW_RISK_WORKERS = Gauge(
    "ppe_low_risk_workers",
    "Number of workers with LOW risk",
)

TOTAL_ALERTS = Gauge(
    "ppe_total_alerts",
    "Total current alerts",
)


# ============================================================
# WORKER METRICS
# ============================================================

WORKER_RISK_SCORE = Gauge(
    "ppe_worker_risk_score",
    "Current risk score for each worker",
    ["worker_id"],
)

WORKER_RISK_LEVEL = Gauge(
    "ppe_worker_risk_level",
    "Current risk level encoded numerically",
    ["worker_id", "risk_level"],
)

WORKER_VIOLATIONS = Gauge(
    "ppe_worker_violations",
    "Current PPE violations for each worker",
    ["worker_id"],
)

WORKER_DECISION = Gauge(
    "ppe_worker_decision",
    "Current supervisor decision encoded numerically",
    ["worker_id", "decision"],
)

WORKER_PPE_STATUS = Gauge(
    "ppe_worker_ppe_status",
    "Current PPE status: 1 compliant, 0 non-compliant",
    ["worker_id", "ppe_type"],
)


# ============================================================
# ENCODING HELPERS
# ============================================================

RISK_LEVEL_VALUES = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

DECISION_VALUES = {
    "NO_ACTION": 0,
    "MONITOR": 1,
    "SEND_WARNING": 2,
    "ESCALATE_IMMEDIATELY": 3,
}


# ============================================================
# UPDATE METRICS FROM SHARED STATE
# ============================================================

def update_metrics(shared_state, event_count):
    """
    Synchronize Prometheus metrics with SharedState.
    """

    workers = shared_state.get_all_workers()

    TOTAL_WORKERS.set(
        len(workers)
    )

    TOTAL_EVENTS.inc(
        max(
            0,
            event_count - TOTAL_EVENTS._value.get(),
        )
    )

    total_violations = 0
    high_risk = 0
    critical_risk = 0
    medium_risk = 0
    low_risk = 0
    total_alerts = 0

    for worker_id, worker in workers.items():

        # ----------------------------------------------------
        # PPE status
        # ----------------------------------------------------

        ppe_items = [
            "helmet",
            "vest",
            "gloves",
            "boots",
            "goggles",
        ]

        for ppe_type in ppe_items:

            status = (
                1
                if worker.get(
                    ppe_type,
                    True,
                )
                else 0
            )

            WORKER_PPE_STATUS.labels(
                worker_id=worker_id,
                ppe_type=ppe_type,
            ).set(status)

        # ----------------------------------------------------
        # Violations
        # ----------------------------------------------------

        violations = (
            shared_state.get_violations(
                worker_id
            )
        )

        violation_count = len(
            violations
        )

        total_violations += (
            violation_count
        )

        WORKER_VIOLATIONS.labels(
            worker_id=worker_id
        ).set(violation_count)

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk = (
            shared_state.get_risk(
                worker_id
            )
        )

        if risk:

            risk_level = risk.get(
                "risk_level",
                "LOW",
            )

            risk_score = risk.get(
                "risk_score",
                0,
            )

            WORKER_RISK_SCORE.labels(
                worker_id=worker_id
            ).set(risk_score)

            WORKER_RISK_LEVEL.labels(
                worker_id=worker_id,
                risk_level=risk_level,
            ).set(
                RISK_LEVEL_VALUES.get(
                    risk_level,
                    0,
                )
            )

            if risk_level == "CRITICAL":
                critical_risk += 1

            elif risk_level == "HIGH":
                high_risk += 1

            elif risk_level == "MEDIUM":
                medium_risk += 1

            else:
                low_risk += 1

        # ----------------------------------------------------
        # Supervisor decision
        # ----------------------------------------------------

        decision = (
            shared_state.get_decision(
                worker_id
            )
        )

        if decision:

            decision_name = decision.get(
                "decision",
                "NO_ACTION",
            )

            WORKER_DECISION.labels(
                worker_id=worker_id,
                decision=decision_name,
            ).set(
                DECISION_VALUES.get(
                    decision_name,
                    0,
                )
            )

        # ----------------------------------------------------
        # Alerts
        # ----------------------------------------------------

        alert = (
            shared_state.get_alert(
                worker_id
            )
        )

        if alert:
            total_alerts += 1

    TOTAL_VIOLATIONS.set(
        total_violations
    )

    HIGH_RISK_WORKERS.set(
        high_risk
    )

    CRITICAL_RISK_WORKERS.set(
        critical_risk
    )

    MEDIUM_RISK_WORKERS.set(
        medium_risk
    )

    LOW_RISK_WORKERS.set(
        low_risk
    )

    TOTAL_ALERTS.set(
        total_alerts
    )


def get_metrics():
    """
    Return Prometheus formatted metrics.
    """

    return generate_latest()