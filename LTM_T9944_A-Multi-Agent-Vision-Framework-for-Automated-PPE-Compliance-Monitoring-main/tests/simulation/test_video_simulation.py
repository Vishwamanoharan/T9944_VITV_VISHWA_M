from src.agents.simulation_agent.simulation_agent import (
    SimulationAgent
)


def test_video_simulation():

    video_path = "data/videos/construction_site.mp4"

    simulator = SimulationAgent()

    simulator.run_video(video_path)


if __name__ == "__main__":
    test_video_simulation()