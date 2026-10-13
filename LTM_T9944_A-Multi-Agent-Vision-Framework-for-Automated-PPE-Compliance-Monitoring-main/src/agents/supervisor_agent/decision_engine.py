class SupervisorDecisionEngine:
    """
    Determines the supervisor action using:
    - Current risk
    - PPE violations
    - Historical violation count
    - Retrieved safety knowledge
    """

    def decide(
        self,
        risk,
        violations,
        history,
        retrieved_knowledge,
    ):

        risk_level = risk.get(
            "risk_level",
            "LOW",
        )

        risk_score = risk.get(
            "risk_score",
            0,
        )

        # -----------------------------------------------------
        # Count historical violations
        # -----------------------------------------------------

        historical_violations = 0

        for event in history:

            if (
                event.get("event_type")
                == "PPE_VIOLATION_DETECTED"
            ):
                historical_violations += 1

        current_violations = len(
            violations
        )

        total_violations = (
            historical_violations
            + current_violations
        )

        # -----------------------------------------------------
        # Decision
        # -----------------------------------------------------

        if risk_level == "CRITICAL":

            decision = (
                "ESCALATE_IMMEDIATELY"
            )

            priority = "CRITICAL"

        elif (
            risk_level == "HIGH"
            or total_violations >= 3
        ):

            decision = (
                "SEND_WARNING"
            )

            priority = "HIGH"

        elif risk_level == "MEDIUM":

            decision = "MONITOR"

            priority = "MEDIUM"

        else:

            decision = "NO_ACTION"

            priority = "LOW"

        # -----------------------------------------------------
        # Knowledge information
        # -----------------------------------------------------

        knowledge_sources = []

        for document in retrieved_knowledge:

            source = document.get(
                "source"
            )

            if source and source not in knowledge_sources:
                knowledge_sources.append(
                    source
                )

        return {
            "decision": decision,
            "priority": priority,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "current_violations": current_violations,
            "historical_violations": historical_violations,
            "total_violations": total_violations,
            "knowledge_sources": knowledge_sources,
        }