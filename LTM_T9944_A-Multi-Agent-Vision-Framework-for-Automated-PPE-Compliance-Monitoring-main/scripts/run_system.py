import cv2

from src.state.state_manager import StateManager

from src.events.event_bus import EventBus

from src.agents.vision_agent.detector import PPEDetector
from src.agents.vision_agent.vision_agent import VisionAgent

from src.agents.tracking_agent.tracker import SimpleTracker
from src.agents.tracking_agent.tracking_agent import TrackingAgent

from src.agents.simulation_agent.scenario_engine import ScenarioEngine
from src.agents.simulation_agent.simulation_agent import SimulationAgent

from src.agents.compliance_agent.rules import ComplianceRules
from src.agents.compliance_agent.compliance_agent import ComplianceAgent

from src.agents.risk_agent.risk_model import RiskModel
from src.agents.risk_agent.risk_agent import RiskAgent

from src.agents.rca_agent.rca_engine import RCAEngine
from src.agents.rca_agent.rca_agent import RCAAgent

from src.agents.recommendation_agent.action_generator import ActionGenerator
from src.agents.recommendation_agent.recommendation_agent import RecommendationAgent

from src.agents.alert_agent.alert_manager import AlertManager
from src.agents.alert_agent.alert_agent import AlertAgent

from src.agents.memory_agent.memory_agent import MemoryAgent

from src.agents.supervisor_agent.workflow import PPEWorkflow
from src.agents.supervisor_agent.supervisor_agent import SupervisorAgent


MODEL_PATH = "models/ppe_yolo.pt"


def build_system():

    event_bus = EventBus()

    state_manager = StateManager()

    # Vision
    detector = PPEDetector(
        MODEL_PATH
    )

    vision_agent = VisionAgent(
        detector,
        event_bus,
        state_manager
    )

    # Tracking
    tracker = SimpleTracker()

    tracking_agent = TrackingAgent(
        tracker,
        event_bus,
        state_manager
    )

    # Simulation
    scenario_engine = ScenarioEngine()

    simulation_agent = SimulationAgent(
        scenario_engine,
        event_bus,
        state_manager
    )

    # Compliance
    compliance_agent = ComplianceAgent(
        ComplianceRules,
        event_bus,
        state_manager
    )

    # Risk
    risk_agent = RiskAgent(
        RiskModel(),
        event_bus,
        state_manager
    )

    # RCA
    rca_agent = RCAAgent(
        RCAEngine(),
        event_bus,
        state_manager
    )

    # Recommendation
    recommendation_agent = RecommendationAgent(
        ActionGenerator(),
        event_bus,
        state_manager
    )

    # Alert
    alert_agent = AlertAgent(
        AlertManager(),
        event_bus,
        state_manager
    )

    # Memory
    memory_agent = MemoryAgent()

    workflow = PPEWorkflow(
        vision_agent,
        tracking_agent,
        simulation_agent,
        compliance_agent,
        risk_agent,
        rca_agent,
        recommendation_agent,
        alert_agent,
        memory_agent
    )

    supervisor = SupervisorAgent(
        workflow
    )

    return supervisor, state_manager


def main():

    supervisor, state_manager = build_system()

    video_path = "data/videos/construction_site.mp4"

    cap = cv2.VideoCapture(video_path)

    frame_id = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        state_manager.start_frame(
            frame_id
        )

        result = supervisor.process_frame(
            frame
        )

        print(
            f"Frame: {frame_id} | "
            f"Status: {result['status']}"
        )

        if result["status"] == "VIOLATION_DETECTED":

            print(
                "Risk:",
                result["risk"]
            )

            print(
                "RCA:",
                result["rca"]
            )

            print(
                "Recommendations:",
                result["recommendations"]
            )

            print(
                "Alerts:",
                result["alerts"]
            )

        cv2.imshow(
            "Multi-Agent PPE Monitoring",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()