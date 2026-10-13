import time

from src.events.event_bus import EventBus
from src.events.event_schema import Event
from src.events.event_types import EventTypes

from src.state.shared_state import SharedState

from src.agents.simulation_agent.scenario_engine import ScenarioEngine
from src.agents.compliance_agent.compliance_agent import ComplianceAgent
from src.agents.risk_agent.risk_agent import RiskAgent
from src.agents.rca_agent.rca_agent import RCAAgent
from src.agents.recommendation_agent.recommendation_agent import RecommendationAgent
from src.agents.alert_agent.alert_agent import AlertAgent
from src.agents.memory_agent.memory_agent import MemoryAgent
from src.agents.supervisor_agent.supervisor_agent import SupervisorAgent


class SimulationAgent:

    def __init__(self):

        # =====================================================
        # Event communication layer
        # =====================================================

        self.event_bus = EventBus()

        # =====================================================
        # Shared system state
        # =====================================================

        self.shared_state = SharedState()

        # =====================================================
        # Agents
        # =====================================================

        self.compliance_agent = ComplianceAgent(
            self.event_bus
        )

        self.risk_agent = RiskAgent(
            self.event_bus
        )

        self.rca_agent = RCAAgent(
            self.event_bus
        )

        self.recommendation_agent = RecommendationAgent(
            self.event_bus
        )

        self.alert_agent = AlertAgent(
            self.event_bus
        )

        self.memory_agent = MemoryAgent(
            self.event_bus,
            self.shared_state,
        )

        self.supervisor_agent = SupervisorAgent(
            self.event_bus,
            self.shared_state,
        )

        # =====================================================
        # Dynamic simulation engine
        # =====================================================

        self.scenario_engine = ScenarioEngine()

        # =====================================================
        # Supervisor decision listener
        # =====================================================

        self.event_bus.subscribe(
            EventTypes.SUPERVISOR_DECISION,
            self.handle_supervisor_decision,
        )

    # =========================================================
    # Supervisor decision
    # =========================================================

    def handle_supervisor_decision(self, event):

        decision = event.payload

        print(
            f"{event.worker_id} | "
            f"Decision={decision['decision']}"
        )

    # =========================================================
    # Initialize workers
    # =========================================================

    def initialize_workers(self, workers_count=3):

        """
        Create the initial dynamic workers.

        This is used by the Streamlit dashboard and can also
        be used by other simulation interfaces.
        """

        workers = self.scenario_engine.create_workers(
            workers_count
        )

        # Publish initial worker state.

        self.event_bus.publish(
            Event(
                event_type=EventTypes.WORKER_STATE_UPDATED,
                source_agent="SimulationAgent",
                payload={
                    "workers": workers
                },
            )
        )

        return workers

    # =========================================================
    # Run ONE dynamic simulation cycle
    # =========================================================

    def run_cycle(self, workers_count=3):

        """
        Execute exactly one dynamic simulation cycle.

        This method is specifically useful for Streamlit,
        where the dashboard needs to update after every cycle.

        Existing Phase 2-8 logic is reused:

        ScenarioEngine
            ↓
        Worker State
            ↓
        Event Bus
            ↓
        Compliance Agent
            ↓
        Risk Agent
            ↓
        RCA Agent
            ↓
        Recommendation Agent
            ↓
        Alert Agent
            ↓
        Memory Agent
            ↓
        Supervisor Agent
        """

        # -----------------------------------------------------
        # Create workers on the first cycle.
        # -----------------------------------------------------

        if not self.scenario_engine.worker_states:

            workers = self.scenario_engine.create_workers(
                workers_count
            )

        # -----------------------------------------------------
        # Generate next dynamic state.
        # -----------------------------------------------------

        workers = self.scenario_engine.next_cycle()

        # -----------------------------------------------------
        # Publish worker state event.
        # -----------------------------------------------------

        self.event_bus.publish(
            Event(
                event_type=EventTypes.WORKER_STATE_UPDATED,
                source_agent="SimulationAgent",
                payload={
                    "workers": workers
                },
            )
        )

        # -----------------------------------------------------
        # Existing Phase-4 event-driven compliance flow.
        # -----------------------------------------------------

        self.compliance_agent.evaluate(
            workers
        )

        return workers

    # =========================================================
    # Run complete simulation
    # =========================================================

    def run(
        self,
        cycles=10,
        workers_count=3,
    ):

        """
        Run the complete batch simulation.

        This method is preserved for the existing Phase 2-8
        simulation and tests.
        """

        # -----------------------------------------------------
        # Create workers.
        # -----------------------------------------------------

        workers = self.scenario_engine.create_workers(
            workers_count
        )

        print("=" * 70)
        print("PHASE 5 - SHARED STATE + AGENT MEMORY")
        print("=" * 70)

        # -----------------------------------------------------
        # Simulation cycles.
        # -----------------------------------------------------

        for cycle in range(1, cycles + 1):

            print()
            print(
                f"--- CYCLE {cycle} ---"
            )

            # Generate next dynamic state.

            workers = self.scenario_engine.next_cycle()

            # -------------------------------------------------
            # Publish current worker state.
            # -------------------------------------------------

            self.event_bus.publish(
                Event(
                    event_type=EventTypes.WORKER_STATE_UPDATED,
                    source_agent="SimulationAgent",
                    payload={
                        "workers": workers
                    },
                )
            )

            # -------------------------------------------------
            # Existing Phase-4 compliance flow.
            # -------------------------------------------------

            self.compliance_agent.evaluate(
                workers
            )

            # -------------------------------------------------
            # Display worker state.
            # -------------------------------------------------

            for worker in workers:

                print(
                    f"{worker['worker_id']} | "
                    f"Event={worker['event']}"
                )

            time.sleep(1)

        # -----------------------------------------------------
        # Simulation summary.
        # -----------------------------------------------------

        print()
        print("=" * 70)
        print("PHASE 5 SIMULATION COMPLETED")
        print("=" * 70)

        print(
            f"Total events processed: "
            f"{len(self.event_bus.get_history())}"
        )

        print(
            f"Workers in shared state: "
            f"{len(self.shared_state.get_all_workers())}"
        )

        # -----------------------------------------------------
        # Worker results.
        # -----------------------------------------------------

        for worker_id in self.shared_state.get_all_workers():

            risk = self.shared_state.get_risk(
                worker_id
            )

            decision = self.shared_state.get_decision(
                worker_id
            )

            print(
                f"{worker_id} | "
                f"Risk="
                f"{risk['risk_level'] if risk else 'N/A'} | "
                f"Decision="
                f"{decision['decision'] if decision else 'N/A'}"
            )