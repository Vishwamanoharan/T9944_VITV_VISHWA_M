from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def main():

    print("=" * 70)
    print("PHASE 8 FASTAPI TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Root
    # --------------------------------------------------------

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"

    print()
    print("Root endpoint: PASSED")

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    print("Health endpoint: PASSED")

    # --------------------------------------------------------
    # Initial workers
    # --------------------------------------------------------

    response = client.get("/workers")

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 0

    print("Initial workers endpoint: PASSED")

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    response = client.post(
        "/simulation/run?cycles=2&workers=3"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["cycles"] == 2
    assert data["workers"] == 3
    assert data["events_processed"] > 0

    print("Simulation endpoint: PASSED")

    # --------------------------------------------------------
    # Workers
    # --------------------------------------------------------

    response = client.get("/workers")

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 3

    print("Workers endpoint: PASSED")

    # --------------------------------------------------------
    # Worker snapshot
    # --------------------------------------------------------

    response = client.get(
        "/workers/Worker-001"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["worker"]["worker_id"] == (
        "Worker-001"
    )

    assert data["risk"] is not None
    assert data["decision"] is not None

    print("Worker snapshot endpoint: PASSED")

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    response = client.get(
        "/workers/Worker-001/risk"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["risk"] is not None

    print("Risk endpoint: PASSED")

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    response = client.get(
        "/workers/Worker-001/decision"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["decision"] is not None

    print("Decision endpoint: PASSED")

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    response = client.get(
        "/workers/Worker-001/history"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["history"]) > 0

    print("History endpoint: PASSED")

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    response = client.get("/state")

    assert response.status_code == 200

    data = response.json()

    assert len(data["workers"]) == 3

    print("State endpoint: PASSED")

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    response = client.get("/events")

    assert response.status_code == 200

    data = response.json()

    assert data["count"] > 0
    assert len(data["events"]) > 0

    print("Events endpoint: PASSED")

    # --------------------------------------------------------
    # Invalid worker
    # --------------------------------------------------------

    response = client.get(
        "/workers/Worker-999"
    )

    assert response.status_code == 404

    print("404 handling: PASSED")

    print()
    print("=" * 70)
    print("PHASE 8 FASTAPI TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()