class RCAEngine:

    def analyze(self, worker, violations):

        worker_id = worker["worker_id"]

        worker_violations = [
            v for v in violations
            if v["worker_id"] == worker_id
        ]

        if not worker_violations:

            return {
                "worker_id": worker_id,
                "root_cause": "NO_VIOLATION",
                "confidence": 1.0
            }

        violation_types = [
            v["violation"]
            for v in worker_violations
        ]

        if len(violation_types) >= 3:

            root_cause = "MULTIPLE_PPE_NON_COMPLIANCE"

        elif "NO_HELMET" in violation_types:

            root_cause = "HELMET_NON_COMPLIANCE"

        elif "NO_VEST" in violation_types:

            root_cause = "VEST_NON_COMPLIANCE"

        else:

            root_cause = "PPE_NON_COMPLIANCE"

        return {
            "worker_id": worker_id,
            "root_cause": root_cause,
            "confidence": 0.85
        }