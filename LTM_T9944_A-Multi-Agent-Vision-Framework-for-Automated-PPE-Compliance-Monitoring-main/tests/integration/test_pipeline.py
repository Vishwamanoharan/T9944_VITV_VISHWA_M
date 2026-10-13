from pathlib import Path

import cv2

from src.agents.vision_agent.vision_agent import VisionAgent
from src.agents.tracking_agent.tracking_agent import TrackingAgent
from src.agents.compliance_agent.compliance_agent import ComplianceAgent


ROOT_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT_DIR / "data" / "images" / "val"


def find_test_image():
    """Find the first available image from the validation dataset."""

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    for image_path in IMAGE_DIR.iterdir():

        if image_path.is_file() and image_path.suffix.lower() in extensions:
            return image_path

    raise FileNotFoundError(
        f"No test images found in: {IMAGE_DIR}"
    )


def main():

    print()
    print("=" * 70)
    print("PPE MULTI-AGENT PIPELINE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Find image
    # ---------------------------------------------------------

    image_path = find_test_image()

    print()
    print(f"Test image: {image_path}")

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------

    frame = cv2.imread(str(image_path))

    if frame is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    print(
        f"Image loaded: "
        f"{frame.shape[1]}x{frame.shape[0]}"
    )

    # ---------------------------------------------------------
    # Initialize agents
    # ---------------------------------------------------------

    print()
    print("Initializing agents...")

    vision_agent = VisionAgent()
    tracking_agent = TrackingAgent()
    compliance_agent = ComplianceAgent()

    print("Vision Agent       : OK")
    print("Tracking Agent     : OK")
    print("Compliance Agent   : OK")

    # ---------------------------------------------------------
    # Vision Agent
    # ---------------------------------------------------------

    print()
    print("[1] Vision Agent")

    vision_output = vision_agent.process_frame(frame)

    detections = vision_output["detections"]

    print(f"Detections: {len(detections)}")

    for detection in detections:

        print(
            f"  {detection['class']:<12} "
            f"confidence={detection['confidence']:.3f} "
            f"bbox={detection['bbox']}"
        )

    # ---------------------------------------------------------
    # Tracking Agent
    # ---------------------------------------------------------

    print()
    print("[2] Tracking Agent")

    workers = tracking_agent.track(
        detections
    )

    print(f"Workers detected: {len(workers)}")

    for worker in workers:

        print(
            f"  {worker['worker_id']} "
            f"bbox={worker['bbox']}"
        )

    # ---------------------------------------------------------
    # Compliance Agent
    # ---------------------------------------------------------

    print()
    print("[3] Compliance Agent")

    violations = compliance_agent.evaluate(
        workers,
        detections
    )

    print(
        f"Violations detected: "
        f"{len(violations)}"
    )

    for violation in violations:

        print(
            f"  {violation['worker_id']} "
            f"→ {violation['violation']} "
            f"→ {violation['severity']}"
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE TEST COMPLETED")
    print("=" * 70)

    print()
    print(f"Image       : {image_path.name}")
    print(f"Detections  : {len(detections)}")
    print(f"Workers     : {len(workers)}")
    print(f"Violations  : {len(violations)}")
    print()


if __name__ == "__main__":
    main()