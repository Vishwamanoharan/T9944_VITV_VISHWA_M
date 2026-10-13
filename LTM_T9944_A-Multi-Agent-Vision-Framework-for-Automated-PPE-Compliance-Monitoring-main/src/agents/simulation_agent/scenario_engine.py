
import random


class ScenarioEngine:
    """
    Generates dynamic worker PPE states for CCTV-like simulation.

    Each simulation cycle represents one monitoring interval.
    """

    EVENTS = [
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NO_HELMET",
        "NO_VEST",
        "NO_GLOVES",
        "NO_BOOTS",
        "NO_GOGGLES",
    ]

    def __init__(self):
        self.worker_states = {}

    def create_workers(self, count=3):
        workers = []

        for index in range(1, count + 1):
            worker = {
                "worker_id": f"Worker-{index:03d}",
                "helmet": True,
                "vest": True,
                "gloves": True,
                "boots": True,
                "goggles": True,
                "event": "NORMAL",
            }

            workers.append(worker)

        self.worker_states = {
            worker["worker_id"]: worker
            for worker in workers
        }

        return workers

    def _apply_event(self, worker, event):
        """
        Reset PPE to compliant state and apply one event.
        """

        worker["helmet"] = True
        worker["vest"] = True
        worker["gloves"] = True
        worker["boots"] = True
        worker["goggles"] = True

        if event == "NO_HELMET":
            worker["helmet"] = False

        elif event == "NO_VEST":
            worker["vest"] = False

        elif event == "NO_GLOVES":
            worker["gloves"] = False

        elif event == "NO_BOOTS":
            worker["boots"] = False

        elif event == "NO_GOGGLES":
            worker["goggles"] = False

        worker["event"] = event

    def next_cycle(self):
        """
        Generate the next dynamic CCTV monitoring cycle.

        Each worker receives an independent event.
        """

        for worker in self.worker_states.values():

            event = random.choice(self.EVENTS)

            self._apply_event(worker, event)

        return list(self.worker_states.values())

