from src.agents.simulation_agent.simulation_agent import SimulationAgent


def main():

    simulation = SimulationAgent()

    simulation.run(
        cycles=10,
        workers_count=3
    )


if __name__ == "__main__":
    main()