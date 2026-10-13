class ActionGenerator:

    def generate(self, risk):

        level = risk["risk_level"]

        if level == "CRITICAL":

            return {
                "action": "STOP_WORK_AND_ESCALATE",
                "priority": "CRITICAL"
            }

        if level == "HIGH":

            return {
                "action": "ISSUE_IMMEDIATE_WARNING",
                "priority": "HIGH"
            }

        if level == "MEDIUM":

            return {
                "action": "ISSUE_SAFETY_REMINDER",
                "priority": "MEDIUM"
            }

        return {
            "action": "CONTINUE_MONITORING",
            "priority": "LOW"
        }