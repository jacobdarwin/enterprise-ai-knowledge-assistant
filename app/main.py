from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.middleware.rate_limit import limiter
from app.api.routes import chat, documents, health, history, metrics, upload
from app.core.config.langsmith_setup import configure_langsmith
from app.core.config.logging_config import configure_logging, get_logger
from app.core.config.settings import get_settings
from app.repositories.database import init_db

configure_logging()
configure_langsmith()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    log.info("app_startup_complete")
    yield
    log.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Enterprise Knowledge Assistant — RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(upload.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(history.router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        log.error("unhandled_exception", path=str(request.url), error=str(exc))
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
