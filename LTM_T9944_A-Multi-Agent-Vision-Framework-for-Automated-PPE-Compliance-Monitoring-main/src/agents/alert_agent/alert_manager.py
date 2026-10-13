class AlertManager:

    def create_alert(
        self,
        worker_id,
        risk_level,
        violations
    ):

        return {
            "worker_id": worker_id,
            "risk_level": risk_level,
            "violations": violations,
            "recipients": [
                "SITE_SUPERVISOR",
                "SAFETY_OFFICER"
            ]
        }