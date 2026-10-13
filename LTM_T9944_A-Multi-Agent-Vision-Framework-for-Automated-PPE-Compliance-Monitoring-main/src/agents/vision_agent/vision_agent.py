from .detector import PPEDetector


class VisionAgent:

    def __init__(self):

        self.detector = PPEDetector()

    def process_frame(self, frame):

        detections = self.detector.detect(frame)

        return {
            "agent": "vision_agent",
            "detections": detections
        }