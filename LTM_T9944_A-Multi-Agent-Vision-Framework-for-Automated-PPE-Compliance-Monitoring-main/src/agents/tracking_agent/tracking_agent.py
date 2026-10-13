from src.agents.tracking_agent.tracker import SimpleTracker


class TrackingAgent:

    def __init__(self):
        self.tracker = SimpleTracker()

    def track(self, detections):

        person_detections = [
            detection
            for detection in detections
            if detection["class"] == "Person"
        ]

        tracked = self.tracker.update(person_detections)

        workers = []

        for worker_id, data in tracked.items():

            workers.append(
                {
                    "worker_id": worker_id,
                    "bbox": data["bbox"],
                    "centroid": data["centroid"]
                }
            )

        return workers