class PPEWorkflow:

    def __init__(
        self,
        vision_agent,
        tracking_agent,
        simulation_agent,
        compliance_agent,
        risk_agent,
        rca_agent,
        recommendation_agent,
        alert_agent,
        memory_agent
    ):

        self.vision_agent = vision_agent
        self.tracking_agent = tracking_agent
        self.simulation_agent = simulation_agent
        self.compliance_agent = compliance_agent
        self.risk_agent = risk_agent
        self.rca_agent = rca_agent
        self.recommendation_agent = recommendation_agent
        self.alert_agent = alert_agent
        self.memory_agent = memory_agent

    def run(self, frame):

        # 1. Vision
        detections = self.vision_agent.process(
            frame
        )

        # 2. Tracking
        tracks = self.tracking_agent.process(
            detections
        )

        # 3. Dynamic simulation
        self.simulation_agent.process(
            tracks
        )

        # 4. Compliance
        compliance = self.compliance_agent.process()

        non_compliant = [
            result
            for result in compliance
            if result["status"] == "NON_COMPLIANT"
        ]

        # Nothing wrong → stop pipeline
        if not non_compliant:

            return {
                "status": "COMPLIANT",
                "next_agents": []
            }

        # 5. Risk analysis
        risk = self.risk_agent.process()

        critical_or_high = [
            item
            for item in risk
            if item["risk_level"] in [
                "HIGH",
                "CRITICAL"
            ]
        ]

        # 6. RCA
        rca = self.rca_agent.process()

        # 7. Recommendations
        recommendations = (
            self.recommendation_agent.process()
        )

        # 8. Alerts only for high risk
        alerts = []

        if critical_or_high:

            alerts = self.alert_agent.process()

        # 9. Memory
        state = self.risk_agent.state_manager.get_state()

        for worker_id, worker in state.workers.items():

            if worker.compliance_status == "NON_COMPLIANT":

                self.memory_agent.remember(
                    worker_id,
                    {
                        "frame_id": state.frame_id,
                        "risk_level": worker.risk_level,
                        "risk_score": worker.risk_score
                    }
                )

        return {
            "status": "VIOLATION_DETECTED",
            "non_compliant": non_compliant,
            "risk": risk,
            "rca": rca,
            "recommendations": recommendations,
            "alerts": alerts
        }