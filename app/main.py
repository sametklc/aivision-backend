"""
AIVision Backend API
A comprehensive AI-powered creative suite API
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger
from collections import defaultdict
from time import time
import sys

from .config import get_settings
from .routes import ai_routes, tools_routes, upload_routes, credit_routes


# ============================================================================
# RATE LIMITING - In-memory rate limiter (use Redis in production for scaling)
# ============================================================================
class RateLimiter:
    """Simple in-memory rate limiter per IP/user"""
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for given key (IP or user_id)"""
        now = time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[key] = [t for t in self.requests[key] if t > minute_ago]

        # Check limit
        if len(self.requests[key]) >= self.requests_per_minute:
            return False

        # Record request
        self.requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests for this minute"""
        now = time()
        minute_ago = now - 60
        self.requests[key] = [t for t in self.requests[key] if t > minute_ago]
        return max(0, self.requests_per_minute - len(self.requests[key]))


# Global rate limiters
general_limiter = RateLimiter(requests_per_minute=100)  # General API
ai_limiter = RateLimiter(requests_per_minute=20)  # AI generation (more expensive)
credit_limiter = RateLimiter(requests_per_minute=30)  # Credit operations

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

# CORS middleware - SECURITY: Restricted origins
# For mobile apps CORS doesn't apply, but this restricts unauthorized web access
_allowed_origins = settings.ALLOWED_ORIGINS.split(",") if settings.ALLOWED_ORIGINS else []
# If no origins configured, only allow mobile apps (no web browser access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins if _allowed_origins else [],  # Empty = no web access allowed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Explicit methods only
    allow_headers=["Authorization", "Content-Type", "X-Device-ID", "X-Request-ID"],  # Explicit headers
)


# Rate limiting middleware - SECURITY: Prevent API abuse
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting based on endpoint type"""
    # Get client identifier (prefer user from auth, fallback to IP)
    client_ip = request.client.host if request.client else "unknown"
    # Try to get user from auth header for more accurate limiting
    auth_header = request.headers.get("Authorization", "")
    client_key = auth_header[:50] if auth_header else client_ip

    path = request.url.path

    # Apply appropriate rate limiter based on endpoint
    if "/api/v1/ai/" in path:
        limiter = ai_limiter
        limit_name = "AI"
    elif "/api/v1/credits/" in path:
        limiter = credit_limiter
        limit_name = "Credit"
    else:
        limiter = general_limiter
        limit_name = "General"

    if not limiter.is_allowed(client_key):
        logger.warning(f"Rate limit exceeded for {client_key} on {path}")
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": "Too many requests",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": f"Rate limit exceeded. Please wait before making more requests.",
                "retry_after": 60,
            },
            headers={"Retry-After": "60"}
        )

    response = await call_next(request)

    # Add rate limit headers
    remaining = limiter.get_remaining(client_key)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = str(limiter.requests_per_minute)

    return response


# Global exception handler - SECURITY: Never expose details in production
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log full error for debugging (server-side only)
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            # SECURITY: Never expose exception details to client
            # Details are logged server-side for debugging
        }
    )


# Include routers
app.include_router(ai_routes.router)
app.include_router(tools_routes.router)
app.include_router(upload_routes.router)
app.include_router(credit_routes.router)


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
