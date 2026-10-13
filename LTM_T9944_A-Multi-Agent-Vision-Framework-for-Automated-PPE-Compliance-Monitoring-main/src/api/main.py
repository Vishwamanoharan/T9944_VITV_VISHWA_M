from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="Multi-Agent PPE Compliance Monitoring API",
    description=(
        "FastAPI interface for the multi-agent "
        "PPE compliance monitoring system."
    ),
    version="1.0.0",
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "system": "Multi-Agent PPE Compliance Monitoring",
        "status": "running",
        "phase": 8,
    }