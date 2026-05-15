from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Multi-Agent Research Platform",
    version="1.0.0"
)

app.include_router(router)