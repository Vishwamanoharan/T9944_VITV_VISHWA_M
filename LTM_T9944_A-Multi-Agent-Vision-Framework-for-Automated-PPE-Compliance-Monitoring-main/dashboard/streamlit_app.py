
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.agents.simulation_agent.simulation_agent import SimulationAgent
from src.events.event_schema import Event
from src.events.event_types import EventTypes


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = PROJECT_ROOT / "models" / "ppe_yolo.pt"
IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "val"

CONFIDENCE = 0.25

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI PPE Safety Monitoring",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .dashboard-subtitle {
        color: #777;
        font-size: 15px;
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 10px;
    }

    .live-box {
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        background-color: rgba(128,128,128,0.06);
        margin-bottom: 10px;
    }

    .worker-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 12px;
        background-color: rgba(128,128,128,0.04);
    }

    .record-card {
        padding: 14px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.22);
        background-color: rgba(128,128,128,0.04);
        margin-bottom: 8px;
    }

    .status-live {
        color: #138a3d;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


model = load_model()


# ============================================================
# LOAD CCTV IMAGES
# ============================================================

@st.cache_data
def load_cctv_images():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"CCTV image directory not found:\n{IMAGE_DIR}"
        )

    images = sorted(
        [
            path
            for path in IMAGE_DIR.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if not images:
        raise RuntimeError(
            f"No CCTV images found in:\n{IMAGE_DIR}"
        )

    return [str(path) for path in images]


cctv_images = load_cctv_images()


# ============================================================
# SESSION STATE
# ============================================================

if "simulation" not in st.session_state:
    st.session_state.simulation = SimulationAgent()

if "running" not in st.session_state:
    st.session_state.running = False

if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0

if "cycle" not in st.session_state:
    st.session_state.cycle = 0

if "previous_people" not in st.session_state:
    st.session_state.previous_people = []

if "worker_next_id" not in st.session_state:
    st.session_state.worker_next_id = 1

if "current_workers" not in st.session_state:
    st.session_state.current_workers = []

if "worker_reports" not in st.session_state:
    st.session_state.worker_reports = {}

if "compliance_records" not in st.session_state:
    st.session_state.compliance_records = []

if "recommendation_records" not in st.session_state:
    st.session_state.recommendation_records = []

if "supervisor_records" not in st.session_state:
    st.session_state.supervisor_records = []

if "rca_records" not in st.session_state:
    st.session_state.rca_records = []

if "alert_records" not in st.session_state:
    st.session_state.alert_records = []


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}


# ============================================================
# TRACKING
# ============================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def center_distance(box_a, box_b):

    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)

    return (
        ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    )


def assign_worker_ids(people):

    previous = st.session_state.previous_people

    assigned = []
    used_previous = set()

    for person in people:

        best_index = None
        best_distance = float("inf")

        for index, previous_person in enumerate(previous):

            if index in used_previous:
                continue

            distance = center_distance(
                person["box"],
                previous_person["box"],
            )

            if distance < best_distance:
                best_distance = distance
                best_index = index

        if (
            best_index is not None
            and best_distance < 150
        ):

            worker_id = previous[
                best_index
            ]["worker_id"]

            used_previous.add(best_index)

        else:

            worker_id = (
                f"Worker-"
                f"{st.session_state.worker_next_id:03d}"
            )

            st.session_state.worker_next_id += 1

        person["worker_id"] = worker_id

        assigned.append(person)

    st.session_state.previous_people = [
        {
            "worker_id": person["worker_id"],
            "box": person["box"],
        }
        for person in assigned
    ]

    return assigned


# ============================================================
# BOX / PPE MATCHING
# ============================================================

def box_area(box):

    x1, y1, x2, y2 = box

    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0

    return (x2 - x1) * (y2 - y1)


def overlap_ratio(small, large):

    area = box_area(small)

    if area <= 0:
        return 0

    return intersection_area(
        small,
        large,
    ) / area


def point_inside(point, box):

    px, py = point
    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def associate_ppe(people, ppe):

    for person in people:

        person_box = person["box"]

        x1, y1, x2, y2 = person_box

        width = x2 - x1
        height = y2 - y1

        head_region = [
            x1 - width * 0.15,
            y1 - height * 0.05,
            x2 + width * 0.15,
            y1 + height * 0.35,
        ]

        upper_region = [
            x1,
            y1 + height * 0.15,
            x2,
            y1 + height * 0.70,
        ]

        lower_region = [
            x1,
            y1 + height * 0.55,
            x2,
            y2,
        ]

        person["helmet"] = False
        person["vest"] = False
        person["gloves"] = False
        person["boots"] = False
        person["goggles"] = False

        person["matched_ppe"] = []

        for item in ppe:

            item_box = item["box"]
            center = box_center(item_box)

            if not point_inside(center, person_box):

                if (
                    overlap_ratio(
                        item_box,
                        person_box,
                    )
                    < 0.05
                ):
                    continue

            name = item["class_name"]
            matched = False

            if name == "helmet":

                if point_inside(
                    center,
                    head_region,
                ):
                    person["helmet"] = True
                    matched = True

            elif name == "vest":

                if point_inside(
                    center,
                    upper_region,
                ):
                    person["vest"] = True
                    matched = True

            elif name == "gloves":

                if point_inside(
                    center,
                    person_box,
                ):
                    person["gloves"] = True
                    matched = True

            elif name == "boots":

                if point_inside(
                    center,
                    lower_region,
                ):
                    person["boots"] = True
                    matched = True

            elif name == "goggles":

                if point_inside(
                    center,
                    head_region,
                ):
                    person["goggles"] = True
                    matched = True

            elif name == "no_helmet":

                if point_inside(
                    center,
                    head_region,
                ):
                    person["helmet"] = False
                    matched = True

            elif name == "no_goggle":

                if point_inside(
                    center,
                    head_region,
                ):
                    person["goggles"] = False
                    matched = True

            elif name == "no_gloves":

                if point_inside(
                    center,
                    person_box,
                ):
                    person["gloves"] = False
                    matched = True

            elif name == "no_boots":

                if point_inside(
                    center,
                    lower_region,
                ):
                    person["boots"] = False
                    matched = True

            if matched:
                person["matched_ppe"].append(item)

    return people


# ============================================================
# LOCAL RISK CALCULATION
# ============================================================

def calculate_risk(worker):

    violations = []
    score = 0

    if not worker["helmet"]:
        score += 40
        violations.append("NO_HELMET")

    if not worker["vest"]:
        score += 30
        violations.append("NO_VEST")

    if not worker["gloves"]:
        score += 10
        violations.append("NO_GLOVES")

    if not worker["boots"]:
        score += 10
        violations.append("NO_BOOTS")

    if not worker["goggles"]:
        score += 10
        violations.append("NO_GOGGLES")

    if score >= 70:

        level = "CRITICAL"

        recommendation = (
            "STOP WORK AND ESCALATE"
        )

    elif score >= 40:

        level = "HIGH"

        recommendation = (
            "ISSUE IMMEDIATE WARNING"
        )

    elif score >= 20:

        level = "MEDIUM"

        recommendation = (
            "ISSUE SAFETY REMINDER"
        )

    else:

        level = "LOW"

        recommendation = (
            "CONTINUE MONITORING"
        )

    return {
        "score": score,
        "level": level,
        "violations": violations,
        "recommendation": recommendation,
    }


# ============================================================
# PROCESS CCTV FRAME
# ============================================================

def process_frame(frame):

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, confidence, class_id in zip(
            boxes,
            confidences,
            classes,
        ):

            detections.append(
                {
                    "box": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": CLASS_NAMES.get(
                        int(class_id),
                        str(class_id),
                    ),
                }
            )

    people = [
        item
        for item in detections
        if item["class_name"] == "Person"
    ]

    ppe = [
        item
        for item in detections
        if item["class_name"] != "Person"
        and item["class_name"] != "none"
    ]

    people = assign_worker_ids(people)

    workers = associate_ppe(
        people,
        ppe,
    )

    for worker in workers:
        worker["risk"] = calculate_risk(worker)

    return workers


# ============================================================
# CREATE CLEAN CCTV DISPLAY
# ============================================================

def create_cctv_display(frame, workers):

    annotated = frame.copy()

    for worker in workers:

        x1, y1, x2, y2 = [
            int(value)
            for value in worker["box"]
        ]

        risk = worker["risk"]

        level = risk["level"]

        if level == "CRITICAL":

            box_color = (
                0,
                0,
                255,
            )

        elif level == "HIGH":

            box_color = (
                0,
                165,
                255,
            )

        elif level == "MEDIUM":

            box_color = (
                0,
                215,
                255,
            )

        else:

            box_color = (
                0,
                200,
                0,
            )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            box_color,
            3,
        )

        label = (
            f"{worker['worker_id']} | "
            f"{level}"
        )

        label_y = max(
            30,
            y1 - 8,
        )

        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        annotated,
        (0, 0),
        (annotated.shape[1], 48),
        (20, 20, 20),
        -1,
    )

    overlay_text = (
        f"CCTV-01  |  "
        f"LIVE  |  "
        f"WORKERS: {len(workers)}"
    )

    cv2.putText(
        annotated,
        overlay_text,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


# ============================================================
# SEND CCTV WORKERS INTO EXISTING MULTI-AGENT SYSTEM
# ============================================================

def send_workers_to_agents(workers):

    if not workers:
        return

    clean_workers = []

    for worker in workers:

        clean_workers.append(
            {
                "worker_id": worker["worker_id"],
                "helmet": bool(worker["helmet"]),
                "vest": bool(worker["vest"]),
                "gloves": bool(worker["gloves"]),
                "boots": bool(worker["boots"]),
                "goggles": bool(worker["goggles"]),
                "event": (
                    "NORMAL"
                    if not worker["risk"]["violations"]
                    else worker["risk"]["violations"][0]
                ),
            }
        )

    event = Event(
        event_type=EventTypes.WORKER_STATE_UPDATED,
        source_agent="CCTVMonitoring",
        payload={
            "workers": clean_workers
        },
    )

    st.session_state.simulation.event_bus.publish(
        event
    )

    st.session_state.simulation.compliance_agent.evaluate(
        clean_workers
    )


# ============================================================
# RECORD AGENT RESULTS
# ============================================================

def record_agent_results(workers):

    shared_state = (
        st.session_state.simulation.shared_state
    )

    now = datetime.now().strftime(
        "%H:%M:%S"
    )

    for worker in workers:

        worker_id = worker["worker_id"]

        risk = worker["risk"]

        violations = risk["violations"]

        # ----------------------------------------------------
        # Compliance record
        # ----------------------------------------------------

        compliance_record = {
            "time": now,
            "frame": st.session_state.frame_index + 1,
            "worker": worker_id,
            "violations": ", ".join(
                violations
            ) if violations else "NONE",
            "status": (
                "NON-COMPLIANT"
                if violations
                else "COMPLIANT"
            ),
        }

        st.session_state.compliance_records.append(
            compliance_record
        )

        # ----------------------------------------------------
        # Read agent results
        # ----------------------------------------------------

        rca = shared_state.get_rca(
            worker_id
        )

        recommendation = (
            shared_state.get_recommendation(
                worker_id
            )
        )

        decision = (
            shared_state.get_decision(
                worker_id
            )
        )

        alert = (
            shared_state.get_alert(
                worker_id
            )
        )

        # ----------------------------------------------------
        # RCA record
        # ----------------------------------------------------

        if rca:

            rca_record = {
                "time": now,
                "worker": worker_id,
                "root_cause": str(
                    rca.get(
                        "root_cause",
                        rca.get(
                            "cause",
                            "PPE_NON_COMPLIANCE",
                        ),
                    )
                ),
            }

            st.session_state.rca_records.append(
                rca_record
            )

        # ----------------------------------------------------
        # Recommendation record
        # ----------------------------------------------------

        if recommendation:

            recommendation_text = (
                recommendation.get(
                    "recommendation",
                    recommendation.get(
                        "action",
                        risk["recommendation"],
                    ),
                )
            )

            recommendation_record = {
                "time": now,
                "worker": worker_id,
                "recommendation": str(
                    recommendation_text
                ),
            }

            st.session_state.recommendation_records.append(
                recommendation_record
            )

        else:

            st.session_state.recommendation_records.append(
                {
                    "time": now,
                    "worker": worker_id,
                    "recommendation": risk[
                        "recommendation"
                    ],
                }
            )

        # ----------------------------------------------------
        # Supervisor record
        # ----------------------------------------------------

        if decision:

            supervisor_record = {
                "time": now,
                "worker": worker_id,
                "risk": decision.get(
                    "risk_level",
                    risk["level"],
                ),
                "score": decision.get(
                    "risk_score",
                    risk["score"],
                ),
                "decision": decision.get(
                    "decision",
                    "MONITOR",
                ),
            }

            st.session_state.supervisor_records.append(
                supervisor_record
            )

        else:

            if risk["level"] == "CRITICAL":
                supervisor_decision = (
                    "ESCALATE_IMMEDIATELY"
                )

            elif risk["level"] == "HIGH":
                supervisor_decision = (
                    "SEND_WARNING"
                )

            elif risk["level"] == "MEDIUM":
                supervisor_decision = (
                    "MONITOR"
                )

            else:
                supervisor_decision = (
                    "NO_ACTION"
                )

            st.session_state.supervisor_records.append(
                {
                    "time": now,
                    "worker": worker_id,
                    "risk": risk["level"],
                    "score": risk["score"],
                    "decision": supervisor_decision,
                }
            )

        # ----------------------------------------------------
        # Alert record
        # ----------------------------------------------------

        if alert:

            st.session_state.alert_records.append(
                {
                    "time": now,
                    "worker": worker_id,
                    "alert": str(alert),
                }
            )

        elif risk["level"] in (
            "HIGH",
            "CRITICAL",
        ):

            st.session_state.alert_records.append(
                {
                    "time": now,
                    "worker": worker_id,
                    "alert": risk[
                        "recommendation"
                    ],
                }
            )


# ============================================================
# WORKER REPORT
# ============================================================

def update_worker_report(workers):

    for worker in workers:

        worker_id = worker["worker_id"]

        risk = worker["risk"]

        if worker_id not in (
            st.session_state.worker_reports
        ):

            st.session_state.worker_reports[
                worker_id
            ] = {
                "worker_id": worker_id,
                "frames_seen": 0,
                "compliant_frames": 0,
                "violation_frames": 0,
                "total_violations": 0,
                "helmet_missing": 0,
                "vest_missing": 0,
                "gloves_missing": 0,
                "boots_missing": 0,
                "goggles_missing": 0,
                "high_risk_frames": 0,
                "critical_frames": 0,
                "last_risk": "LOW",
                "last_score": 0,
                "last_decision": "NO_ACTION",
                "last_recommendation": (
                    "CONTINUE MONITORING"
                ),
                "last_seen_frame": 0,
            }

        report = (
            st.session_state.worker_reports[
                worker_id
            ]
        )

        report["frames_seen"] += 1

        report["last_seen_frame"] = (
            st.session_state.frame_index + 1
        )

        report["last_risk"] = (
            risk["level"]
        )

        report["last_score"] = (
            risk["score"]
        )

        report["last_recommendation"] = (
            risk["recommendation"]
        )

        if risk["violations"]:

            report["violation_frames"] += 1

            report["total_violations"] += len(
                risk["violations"]
            )

        else:

            report["compliant_frames"] += 1

        if not worker["helmet"]:
            report["helmet_missing"] += 1

        if not worker["vest"]:
            report["vest_missing"] += 1

        if not worker["gloves"]:
            report["gloves_missing"] += 1

        if not worker["boots"]:
            report["boots_missing"] += 1

        if not worker["goggles"]:
            report["goggles_missing"] += 1

        if risk["level"] == "HIGH":
            report["high_risk_frames"] += 1

        if risk["level"] == "CRITICAL":
            report["critical_frames"] += 1


# ============================================================
# RESET
# ============================================================

def reset_all():

    st.session_state.simulation = SimulationAgent()

    st.session_state.running = False

    st.session_state.frame_index = 0

    st.session_state.cycle = 0

    st.session_state.previous_people = []

    st.session_state.worker_next_id = 1

    st.session_state.current_workers = []

    st.session_state.worker_reports = {}

    st.session_state.compliance_records = []

    st.session_state.recommendation_records = []

    st.session_state.supervisor_records = []

    st.session_state.rca_records = []

    st.session_state.alert_records = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    '🦺 AI Multi-Agent PPE Safety Monitoring'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Live CCTV worker monitoring, PPE compliance recording, '
    'risk assessment, RCA, recommendations and supervisor decisions.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📹 CCTV Control")

    if st.button(
        "▶ Start Monitoring",
        use_container_width=True,
    ):

        st.session_state.running = True
        st.rerun()

    if st.button(
        "⏹ Stop Monitoring",
        use_container_width=True,
    ):

        st.session_state.running = False
        st.rerun()

    if st.button(
        "🔄 Reset Monitoring",
        use_container_width=True,
    ):

        reset_all()
        st.rerun()

    st.divider()

    st.write(
        "**Camera:** CCTV-01"
    )

    st.write(
        "**Available Frames:**",
        len(cctv_images),
    )

    st.write(
        "**Current Frame:**",
        st.session_state.frame_index + 1,
    )

    st.write(
        "**Monitoring Cycle:**",
        st.session_state.cycle,
    )

    if st.session_state.running:

        st.success(
            "● LIVE MONITORING"
        )

    else:

        st.info(
            "Monitoring stopped"
        )


# ============================================================
# CURRENT CCTV FRAME
# ============================================================

current_index = (
    st.session_state.frame_index
    % len(cctv_images)
)

current_image_path = cctv_images[
    current_index
]

frame = cv2.imread(
    current_image_path
)

if frame is None:

    st.error(
        f"Unable to read CCTV frame:\n"
        f"{current_image_path}"
    )

    st.stop()


# ============================================================
# PROCESS CCTV
# ============================================================

current_workers = process_frame(
    frame
)

st.session_state.current_workers = (
    current_workers
)


# ============================================================
# SEND ACTUAL CCTV WORKERS TO AGENTS
# ============================================================

if current_workers:

    try:

        send_workers_to_agents(
            current_workers
        )

        record_agent_results(
            current_workers
        )

    except Exception as error:

        st.warning(
            f"Agent processing warning: {error}"
        )

    update_worker_report(
        current_workers
    )


# ============================================================
# CREATE CCTV DISPLAY
# ============================================================

display_frame = create_cctv_display(
    frame,
    current_workers,
)


# ============================================================
# TOP LIVE STATUS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔴 Live Site Monitoring'
    '</div>',
    unsafe_allow_html=True,
)

status1, status2, status3, status4, status5 = (
    st.columns(5)
)

current_violations = sum(
    len(worker["risk"]["violations"])
    for worker in current_workers
)

current_high = sum(
    1
    for worker in current_workers
    if worker["risk"]["level"] == "HIGH"
)

current_critical = sum(
    1
    for worker in current_workers
    if worker["risk"]["level"] == "CRITICAL"
)

with status1:
    st.metric(
        "Active Workers",
        len(current_workers),
    )

with status2:
    st.metric(
        "PPE Violations",
        current_violations,
    )

with status3:
    st.metric(
        "High Risk",
        current_high,
    )

with status4:
    st.metric(
        "Critical",
        current_critical,
    )

with status5:
    st.metric(
        "Workers Recorded",
        len(
            st.session_state.worker_reports
        ),
    )


# ============================================================
# LIVE CCTV
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📹 CCTV-01 — Live Worker Monitoring'
    '</div>',
    unsafe_allow_html=True,
)

cctv_col, info_col = st.columns(
    [3.5, 1]
)

with cctv_col:

    st.image(
        cv2.cvtColor(
            display_frame,
            cv2.COLOR_BGR2RGB,
        ),
        use_container_width=True,
    )

    st.caption(
        f"Live source: "
        f"{Path(current_image_path).name}"
    )


with info_col:

    st.markdown(
        '<div class="live-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### 📡 Camera Status"
    )

    st.markdown(
        '<span class="status-live">'
        '● LIVE'
        '</span>',
        unsafe_allow_html=True,
    )

    st.write(
        "Frame:",
        st.session_state.frame_index + 1,
    )

    st.write(
        "Workers:",
        len(current_workers),
    )

    st.write(
        "Monitoring:",
        "ACTIVE"
        if st.session_state.running
        else "PAUSED",
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# LIVE WORKER COMPLIANCE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '👷 Live Worker Compliance Record'
    '</div>',
    unsafe_allow_html=True,
)

if not current_workers:

    st.warning(
        "No workers detected in the current CCTV frame."
    )

else:

    compliance_rows = []

    for worker in current_workers:

        risk = worker["risk"]

        compliance_rows.append(
            {
                "Worker": worker["worker_id"],
                "Helmet": (
                    "COMPLIANT"
                    if worker["helmet"]
                    else "MISSING"
                ),
                "Vest": (
                    "COMPLIANT"
                    if worker["vest"]
                    else "MISSING"
                ),
                "Gloves": (
                    "COMPLIANT"
                    if worker["gloves"]
                    else "MISSING"
                ),
                "Boots": (
                    "COMPLIANT"
                    if worker["boots"]
                    else "MISSING"
                ),
                "Goggles": (
                    "COMPLIANT"
                    if worker["goggles"]
                    else "MISSING"
                ),
                "Violations": (
                    ", ".join(
                        risk["violations"]
                    )
                    if risk["violations"]
                    else "NONE"
                ),
                "Risk": risk["level"],
                "Score": risk["score"],
            }
        )

    compliance_df = pd.DataFrame(
        compliance_rows
    )

    st.dataframe(
        compliance_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CURRENT WORKER DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔎 Current Worker Status'
    '</div>',
    unsafe_allow_html=True,
)

for worker in current_workers:

    risk = worker["risk"]

    with st.container(
        border=True
    ):

        title_col, risk_col = st.columns(
            [3, 1]
        )

        with title_col:

            st.markdown(
                f"### {worker['worker_id']}"
            )

        with risk_col:

            if risk["level"] == "CRITICAL":
                st.error(
                    f"CRITICAL — {risk['score']}"
                )

            elif risk["level"] == "HIGH":
                st.warning(
                    f"HIGH — {risk['score']}"
                )

            elif risk["level"] == "MEDIUM":
                st.info(
                    f"MEDIUM — {risk['score']}"
                )

            else:
                st.success(
                    f"LOW — {risk['score']}"
                )

        p1, p2, p3, p4, p5 = st.columns(5)

        with p1:
            st.write(
                "Helmet",
                "✅"
                if worker["helmet"]
                else "❌",
            )

        with p2:
            st.write(
                "Vest",
                "✅"
                if worker["vest"]
                else "❌",
            )

        with p3:
            st.write(
                "Gloves",
                "✅"
                if worker["gloves"]
                else "❌",
            )

        with p4:
            st.write(
                "Boots",
                "✅"
                if worker["boots"]
                else "❌",
            )

        with p5:
            st.write(
                "Goggles",
                "✅"
                if worker["goggles"]
                else "❌",
            )

        if risk["violations"]:

            st.error(
                "Violations: "
                + ", ".join(
                    risk["violations"]
                )
            )

        else:

            st.success(
                "All required PPE compliant."
            )


# ============================================================
# COMPLIANCE RECORD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📝 Compliance Record'
    '</div>',
    unsafe_allow_html=True,
)

if st.session_state.compliance_records:

    compliance_history_df = pd.DataFrame(
        st.session_state.compliance_records[
            -50:
        ]
    )

    st.dataframe(
        compliance_history_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Compliance records will appear during monitoring."
    )


# ============================================================
# RCA + RECOMMENDATION + SUPERVISOR
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🧠 Safety Analysis and Supervisor Actions'
    '</div>',
    unsafe_allow_html=True,
)

analysis1, analysis2, analysis3 = st.columns(
    3
)


# ============================================================
# RCA
# ============================================================

with analysis1:

    st.markdown(
        "### 🔍 Root Cause Analysis"
    )

    displayed_rca = False

    for worker in current_workers:

        worker_id = worker["worker_id"]

        rca = (
            st.session_state.simulation
            .shared_state
            .get_rca(worker_id)
        )

        if rca:

            displayed_rca = True

            root_cause = rca.get(
                "root_cause",
                rca.get(
                    "cause",
                    "PPE_NON_COMPLIANCE",
                ),
            )

            st.markdown(
                f"**{worker_id}**"
            )

            st.write(
                str(root_cause)
            )

    if not displayed_rca:

        for worker in current_workers:

            if worker["risk"]["violations"]:

                displayed_rca = True

                st.markdown(
                    f"**{worker['worker_id']}**"
                )

                st.write(
                    "PPE_NON_COMPLIANCE"
                )

    if not displayed_rca:

        st.success(
            "No active root cause."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

with analysis2:

    st.markdown(
        "### 💡 Recommendations"
    )

    if current_workers:

        for worker in current_workers:

            worker_id = worker["worker_id"]

            recommendation = (
                st.session_state.simulation
                .shared_state
                .get_recommendation(
                    worker_id
                )
            )

            if recommendation:

                text_value = recommendation.get(
                    "recommendation",
                    recommendation.get(
                        "action",
                        worker["risk"][
                            "recommendation"
                        ],
                    ),
                )

            else:

                text_value = worker["risk"][
                    "recommendation"
                ]

            st.markdown(
                f"**{worker_id}**"
            )

            if worker["risk"]["level"] in (
                "CRITICAL",
                "HIGH",
            ):

                st.warning(
                    str(text_value)
                )

            elif worker["risk"]["level"] == "MEDIUM":

                st.info(
                    str(text_value)
                )

            else:

                st.success(
                    str(text_value)
                )

    else:

        st.info(
            "No worker recommendation available."
        )


# ============================================================
# SUPERVISOR
# ============================================================

with analysis3:

    st.markdown(
        "### 🤖 Supervisor Record"
    )

    if current_workers:

        for worker in current_workers:

            worker_id = worker["worker_id"]

            decision = (
                st.session_state.simulation
                .shared_state
                .get_decision(
                    worker_id
                )
            )

            st.markdown(
                f"**{worker_id}**"
            )

            if decision:

                decision_text = decision.get(
                    "decision",
                    "MONITOR",
                )

                risk_text = decision.get(
                    "risk_level",
                    worker["risk"]["level"],
                )

                st.write(
                    "Risk:",
                    risk_text,
                )

                if risk_text == "CRITICAL":

                    st.error(
                        decision_text
                    )

                elif risk_text == "HIGH":

                    st.warning(
                        decision_text
                    )

                else:

                    st.info(
                        decision_text
                    )

            else:

                st.info(
                    "MONITOR"
                )

    else:

        st.info(
            "No supervisor decision available."
        )


# ============================================================
# SUPERVISOR DECISION HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🤖 Supervisor Decision Record'
    '</div>',
    unsafe_allow_html=True,
)

if st.session_state.supervisor_records:

    supervisor_df = pd.DataFrame(
        st.session_state.supervisor_records[
            -50:
        ]
    )

    st.dataframe(
        supervisor_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Supervisor decisions will appear here."
    )


# ============================================================
# RECOMMENDATION HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '💡 Recommendation Record'
    '</div>',
    unsafe_allow_html=True,
)

if st.session_state.recommendation_records:

    recommendation_df = pd.DataFrame(
        st.session_state.recommendation_records[
            -50:
        ]
    )

    st.dataframe(
        recommendation_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Recommendations will appear here."
    )


# ============================================================
# ACTIVE ALERTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🚨 Active Safety Alerts'
    '</div>',
    unsafe_allow_html=True,
)

active_alerts = [
    worker
    for worker in current_workers
    if worker["risk"]["level"]
    in (
        "HIGH",
        "CRITICAL",
    )
]

if active_alerts:

    for worker in active_alerts:

        if worker["risk"]["level"] == "CRITICAL":

            st.error(
                f"{worker['worker_id']} — "
                f"CRITICAL — "
                f"{', '.join(worker['risk']['violations'])}"
            )

        else:

            st.warning(
                f"{worker['worker_id']} — "
                f"HIGH RISK — "
                f"{', '.join(worker['risk']['violations'])}"
            )

else:

    st.success(
        "No active high-risk safety alerts."
    )


# ============================================================
# WORKER REPORTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📋 Worker Compliance Reports'
    '</div>',
    unsafe_allow_html=True,
)

history = (
    st.session_state.worker_reports
)

if not history:

    st.info(
        "Worker reports will be generated automatically "
        "as workers appear in CCTV frames."
    )

else:

    report_rows = []

    for worker_id, report in history.items():

        frames_seen = report[
            "frames_seen"
        ]

        compliant_frames = report[
            "compliant_frames"
        ]

        if frames_seen > 0:

            compliance = round(
                (
                    compliant_frames
                    /
                    frames_seen
                )
                * 100,
                1,
            )

        else:

            compliance = 100.0

        report_rows.append(
            {
                "Worker": worker_id,
                "Frames Seen": frames_seen,
                "Compliance %": compliance,
                "Total Violations": report[
                    "total_violations"
                ],
                "Helmet Missing": report[
                    "helmet_missing"
                ],
                "Vest Missing": report[
                    "vest_missing"
                ],
                "Gloves Missing": report[
                    "gloves_missing"
                ],
                "Boots Missing": report[
                    "boots_missing"
                ],
                "Goggles Missing": report[
                    "goggles_missing"
                ],
                "High Risk Frames": report[
                    "high_risk_frames"
                ],
                "Critical Frames": report[
                    "critical_frames"
                ],
                "Last Risk": report[
                    "last_risk"
                ],
                "Last Score": report[
                    "last_score"
                ],
                "Last Recommendation": report[
                    "last_recommendation"
                ],
            }
        )

    report_df = pd.DataFrame(
        report_rows
    )

    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # CSV DOWNLOAD
    # --------------------------------------------------------

    csv_data = report_df.to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download Worker Report CSV",
        data=csv_data,
        file_name=(
            "ppe_worker_compliance_report.csv"
        ),
        mime="text/csv",
        use_container_width=False,
    )


# ============================================================
# INDIVIDUAL WORKER REPORT
# ============================================================

if history:

    st.markdown(
        '<div class="section-title">'
        '👤 Individual Worker Report'
        '</div>',
        unsafe_allow_html=True,
    )

    selected_worker = st.selectbox(
        "Select Worker",
        list(history.keys()),
    )

    report = history[
        selected_worker
    ]

    frames_seen = report[
        "frames_seen"
    ]

    compliant_frames = report[
        "compliant_frames"
    ]

    if frames_seen > 0:

        compliance = round(
            (
                compliant_frames
                /
                frames_seen
            )
            * 100,
            1,
        )

    else:

        compliance = 100.0

    r1, r2, r3, r4 = st.columns(4)

    with r1:

        st.metric(
            "Frames Seen",
            frames_seen,
        )

    with r2:

        st.metric(
            "Compliance",
            f"{compliance}%",
        )

    with r3:

        st.metric(
            "High Risk Frames",
            report[
                "high_risk_frames"
            ],
        )

    with r4:

        st.metric(
            "Critical Frames",
            report[
                "critical_frames"
            ],
        )

    detail1, detail2, detail3 = st.columns(
        3
    )

    with detail1:

        st.write(
            "**Helmet Missing:**",
            report[
                "helmet_missing"
            ],
        )

        st.write(
            "**Vest Missing:**",
            report[
                "vest_missing"
            ],
        )

        st.write(
            "**Gloves Missing:**",
            report[
                "gloves_missing"
            ],
        )

    with detail2:

        st.write(
            "**Boots Missing:**",
            report[
                "boots_missing"
            ],
        )

        st.write(
            "**Goggles Missing:**",
            report[
                "goggles_missing"
            ],
        )

        st.write(
            "**Total Violations:**",
            report[
                "total_violations"
            ],
        )

    with detail3:

        st.write(
            "**Last Risk:**",
            report[
                "last_risk"
            ],
        )

        st.write(
            "**Last Score:**",
            report[
                "last_score"
            ],
        )

        st.write(
            "**Last Recommendation:**",
            report[
                "last_recommendation"
            ],
        )


# ============================================================
# SYSTEM RECORD SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Monitoring Records'
    '</div>',
    unsafe_allow_html=True,
)

record1, record2, record3, record4 = (
    st.columns(4)
)

with record1:

    st.metric(
        "Compliance Records",
        len(
            st.session_state.compliance_records
        ),
    )

with record2:

    st.metric(
        "RCA Records",
        len(
            st.session_state.rca_records
        ),
    )

with record3:

    st.metric(
        "Recommendations",
        len(
            st.session_state.recommendation_records
        ),
    )

with record4:

    st.metric(
        "Supervisor Decisions",
        len(
            st.session_state.supervisor_records
        ),
    )


# ============================================================
# LIVE EVENT RECORD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📡 Recent System Records'
    '</div>',
    unsafe_allow_html=True,
)

events = (
    st.session_state.simulation
    .event_bus
    .get_history()
)

if events:

    event_rows = []

    for event in events[-30:]:

        event_type = getattr(
            event,
            "event_type",
            "UNKNOWN",
        )

        source = getattr(
            event,
            "source_agent",
            "UNKNOWN",
        )

        worker_id = getattr(
            event,
            "worker_id",
            None,
        )

        priority = getattr(
            event,
            "priority",
            "LOW",
        )

        timestamp = getattr(
            event,
            "timestamp",
            None,
        )

        if isinstance(
            timestamp,
            datetime,
        ):

            timestamp_text = timestamp.strftime(
                "%H:%M:%S"
            )

        else:

            timestamp_text = str(
                timestamp
            )

        event_rows.append(
            {
                "Time": timestamp_text,
                "Event": event_type,
                "Source": source,
                "Worker": (
                    worker_id
                    or "SYSTEM"
                ),
                "Priority": priority,
            }
        )

    event_df = pd.DataFrame(
        reversed(event_rows)
    )

    st.dataframe(
        event_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No system events recorded yet."
    )


# ============================================================
# AUTOMATIC LIVE MONITORING
# ============================================================

if st.session_state.running:

    st.session_state.cycle += 1

    st.session_state.frame_index = (
        st.session_state.frame_index + 1
    ) % len(cctv_images)

    time.sleep(1.5)

    st.rerun()

