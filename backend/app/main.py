"""
AIVision Backend API
A comprehensive AI-powered creative suite API
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger
import sys

from .config import get_settings
from .routes import ai_routes, tools_routes, upload_routes

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Startup
    if not settings.REPLICATE_API_TOKEN:
        logger.warning("REPLICATE_API_TOKEN not set - AI features will not work!")

    yield

    # Shutdown
    logger.info("Shutting down AIVision API")


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    AIVision API - Premium AI Creative Suite

    ## Features
    - **Video AI**: Text-to-video, image animation, face swap, talking avatars
    - **Photo Enhance**: 4K upscaling, face clarification, denoising, colorization
    - **Magic Edit**: Background removal, object removal, style transfer, age transform

    ## Authentication
    Include your API key in the `X-API-Key` header for authenticated requests.

    ## Rate Limits
    - Free tier: 100 requests/minute
    - Pro tier: 1000 requests/minute
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": str(exc) if settings.DEBUG else None,
        }
    )


# Include routers
app.include_router(ai_routes.router)
app.include_router(tools_routes.router)
app.include_router(upload_routes.router)


# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "replicate_configured": bool(settings.REPLICATE_API_TOKEN),
        "cloudinary_configured": bool(settings.CLOUDINARY_CLOUD_NAME),
    }


# API info endpoint
@app.get("/api/v1/info", tags=["Info"])
async def api_info():
    """Get API information and capabilities."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "capabilities": {
            "video_ai": [
                "text_to_video",
                "image_to_video",
                "script_to_video",
                "ai_hug",
                "ai_kiss",
                "face_swap_video",
                "talking_avatar",
                "lip_sync",
            ],
            "photo_enhance": [
                "4k_upscale",
                "face_clarify",
                "ai_enhance",
                "denoise",
                "colorize",
                "old_photo_restore",
                "remove_scratch",
                "light_adjust",
            ],
            "magic_edit": [
                "background_remove",
                "background_change",
                "ai_extend",
                "object_remove",
                "smart_crop",
                "face_swap",
                "sketch_to_image",
                "style_transfer",
                "ai_filter",
                "age_transform",
                "expression_edit",
                "hair_style",
                "makeup_transfer",
            ],
        },
        "endpoints": {
            "tools": "/api/v1/tools",
            "generate": "/api/v1/ai/generate",
            "job_status": "/api/v1/ai/job/{job_id}",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
