class ComplianceRules:

    @staticmethod
    def evaluate(worker):

        violations = []

        if not worker.helmet_detected:

            violations.append(
                "MISSING_HELMET"
            )

        if not worker.vest_detected:

            violations.append(
                "MISSING_VEST"
            )

        if worker.zone == "DANGER":

            violations.append(
                "RESTRICTED_ZONE_ENTRY"
            )

        if violations:

            return {
                "status": "NON_COMPLIANT",
                "violations": violations
            }

        return {
            "status": "COMPLIANT",
            "violations": []
        }