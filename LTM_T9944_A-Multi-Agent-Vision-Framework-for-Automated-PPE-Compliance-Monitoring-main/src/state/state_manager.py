from typing import Dict, Any
from .shared_state import SharedState, WorkerState


class StateManager:

    def __init__(self):
        self.state = SharedState()

    def start_frame(self, frame_id: int):
        self.state.frame_id = frame_id
        self.state.reset_frame()

    def add_worker(
        self,
        worker_id: str,
        bbox=None
    ):
        if worker_id not in self.state.workers:

            self.state.workers[worker_id] = WorkerState(
                worker_id=worker_id,
                bbox=bbox or []
            )

        else:

            if bbox is not None:
                self.state.workers[worker_id].bbox = bbox

        return self.state.workers[worker_id]

    def get_worker(self, worker_id: str):

        return self.state.workers.get(worker_id)

    def update_worker(
        self,
        worker_id: str,
        **kwargs
    ):

        worker = self.add_worker(worker_id)

        for key, value in kwargs.items():

            if hasattr(worker, key):
                setattr(worker, key, value)

        return worker

    def get_state(self):

        return self.state

    def get_snapshot(self) -> Dict[str, Any]:

        return {
            "frame_id": self.state.frame_id,
            "timestamp": self.state.timestamp,
            "workers": {
                worker_id: vars(worker)
                for worker_id, worker in self.state.workers.items()
            },
            "detections": self.state.detections,
            "simulation_events": self.state.simulation_events,
            "compliance_results": self.state.compliance_results,
            "risk_results": self.state.risk_results,
            "rca_results": self.state.rca_results,
            "recommendations": self.state.recommendations,
            "alerts": self.state.alerts,
            "system_status": self.state.system_status
        }