from ultralytics import YOLO


class PPEDetector:

    def __init__(self, model_path="models/ppe_yolo.pt"):

        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model(frame)

        detections = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                cls_id = int(box.cls[0])

                detections.append(
                    {
                        "class": self.model.names[cls_id],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()
                    }
                )

        return detections