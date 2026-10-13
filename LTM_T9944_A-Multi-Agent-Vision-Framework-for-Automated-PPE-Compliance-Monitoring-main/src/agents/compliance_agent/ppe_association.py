def center(bbox):
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def point_inside_box(point, bbox):

    x, y = point

    x1, y1, x2, y2 = bbox

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


class PPEAssociation:

    PPE_CLASSES = {
        "helmet",
        "vest",
        "gloves",
        "boots",
        "goggles"
    }

    def associate(self, workers, detections):

        worker_ppe = {}

        for worker in workers:

            worker_id = worker["worker_id"]
            worker_bbox = worker["bbox"]

            worker_ppe[worker_id] = {
                "worker_id": worker_id,
                "helmet": False,
                "vest": False,
                "gloves": False,
                "boots": False,
                "goggles": False
            }

            for detection in detections:

                ppe_class = detection["class"]

                if ppe_class not in self.PPE_CLASSES:
                    continue

                ppe_center = center(
                    detection["bbox"]
                )

                if point_inside_box(
                    ppe_center,
                    worker_bbox
                ):

                    worker_ppe[worker_id][ppe_class] = True

        return list(worker_ppe.values())