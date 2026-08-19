from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat import (
    router as chat_router,
)

from api.routes.health import (
    router as health_router,
)

from api.routes.search import (
    router as search_router,
)

from api.routes.graph import (
    router as graph_router,
)

from api.routes.data import (
    router as data_router,
)

from api.routes.evaluation import (
    router as evaluation_router,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print(
        "SickleGuide API starting...",
        flush=True,
    )

    yield

    print(
        "SickleGuide API shutting down...",
        flush=True,
    )


app = FastAPI(
    title="SickleGuide API",
    description=(
        "Evidence-grounded Graph RAG API "
        "for sickle cell disease information."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    health_router,
    prefix="/api/v1",
)

app.include_router(
    chat_router,
    prefix="/api/v1",
)

app.include_router(
    search_router,
    prefix="/api/v1",
)

app.include_router(
    graph_router,
    prefix="/api/v1",
)

app.include_router(
    data_router,
    prefix="/api/v1",
)

app.include_router(
    evaluation_router,
    prefix="/api/v1",
)


@app.get("/")
def root():

    return {
        "service": "SickleGuide",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "graph": "/api/v1/graph",
        "data": "/api/v1/data",
        "evaluation": "/api/v1/evaluation/run",
    }