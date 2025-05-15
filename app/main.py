import logging
import concurrent.futures

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.api.v1.endpoints import cheques as cheques_router
# from app.services.cheque_processing_service import ChequeProcessingService # To init executor early

# Configure logging at the very beginning
configure_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
    # lifespan=lifespan # For managing resources like executor shutdown
)

# Global ThreadPoolExecutor (alternative to service-level)
# Consider managing its lifecycle (startup/shutdown) using FastAPI's lifespan events
# app_executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
# logger.info(f"Global ThreadPoolExecutor initialized with max_workers={settings.MAX_WORKERS}")

# @app.on_event("shutdown")
# async def app_shutdown():
#     logger.info("Shutting down global ThreadPoolExecutor.")
#     app_executor.shutdown(wait=True)
#     logger.info("Global ThreadPoolExecutor shut down complete.")


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Include API routers
app.include_router(cheques_router.router, prefix="/api/v1/cheques", tags=["Cheque Processing"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to the {settings.APP_TITLE}!"}

# Vertex AI initialization is handled by VertexAIService constructor when it's first instantiated.
# No explicit global init here unless strictly necessary for other parts.

logger.info(f"{settings.APP_TITLE} application startup complete.")

# Note: The original `uvicorn.run` in `if __name__ == "__main__":` block is typically
# moved to a separate `run.py` or managed by Docker for production.