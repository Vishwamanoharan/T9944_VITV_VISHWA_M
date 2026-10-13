class RiskModel:

    def calculate(self, worker):

        score = 0

        if not worker.get("helmet", True):
            score += 40

        if not worker.get("vest", True):
            score += 30

        if not worker.get("gloves", True):
            score += 10

        if not worker.get("boots", True):
            score += 10

        if not worker.get("goggles", True):
            score += 10

        if score >= 70:
            level = "CRITICAL"

        elif score >= 40:
            level = "HIGH"

        elif score >= 20:
            level = "MEDIUM"

        else:
            level = "LOW"

        return {
            "risk_score": score,
            "risk_level": level
        }