"""
AIVision API - AI Routes
Endpoints for all AI tools
SECURED with Firebase Auth + Credit Validation
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Header
from typing import Dict, Any
from loguru import logger
import uuid
from datetime import datetime
import firebase_admin
from firebase_admin import auth, firestore, credentials
import os
import json

from ..models.schemas import (
    GenericAIRequest,
    TextToVideoRequest,
    ImageToVideoRequest,
    AIHugRequest,
    FaceSwapVideoRequest,
    TalkingAvatarRequest,
    UpscaleRequest,
    FaceClarifyRequest,
    DenoiseRequest,
    ColorizeRequest,
    OldPhotoRestoreRequest,
    BackgroundRemoveRequest,
    BackgroundChangeRequest,
    ObjectRemoveRequest,
    FaceSwapRequest,
    SketchToImageRequest,
    StyleTransferRequest,
    AgeTransformRequest,
    ExpressionEditRequest,
    JobResponse,
    JobStatusResponse,
    JobStatus,
    ErrorResponse,
)
from ..services.replicate_service import replicate_service
from ..config import AI_MODELS

router = APIRouter(prefix="/api/v1/ai", tags=["AI Tools"])


# ==================== FIREBASE AUTH ====================

def _ensure_firebase_initialized():
    """Ensure Firebase Admin SDK is initialized."""
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_creds_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

        if firebase_creds_json:
            try:
                creds_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(creds_dict)
                bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET", "aivision-47fb4.firebasestorage.app")
                firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
                logger.info("Firebase Admin SDK initialized for AI routes")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase: {e}")
                raise
        else:
            service_account_path = os.environ.get(
                "FIREBASE_SERVICE_ACCOUNT_PATH",
                "firebase-service-account.json"
            )
            if os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET", "aivision-47fb4.firebasestorage.app")
                firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
                logger.info(f"Firebase Admin SDK initialized from file: {service_account_path}")
            else:
                raise ValueError("Firebase credentials not found")


def get_firestore_client():
    """Get Firestore client, initializing Firebase if needed."""
    _ensure_firebase_initialized()
    return firestore.client()


async def verify_firebase_token(authorization: str = Header(...)) -> str:
    """
    Verify Firebase ID Token and return user_id
    Header format: "Bearer <token>"
    """
    _ensure_firebase_initialized()

    if not authorization.startswith("Bearer "):
        logger.warning("🚫 Invalid authorization header format")
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.replace("Bearer ", "")

    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        logger.info(f"🔐 AI Routes - Token verified for user: {user_id}")
        return user_id
    except Exception as e:
        logger.error(f"❌ AI Routes - Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def check_and_deduct_credits(user_id: str, tool_id: str, credit_cost: int) -> bool:
    """
    Check if user has enough credits and deduct them ATOMICALLY using Firestore transaction.
    This prevents race conditions where multiple requests could bypass credit check.
    Returns True if successful, raises HTTPException if not enough credits.
    """
    db = get_firestore_client()
    user_ref = db.collection('users').document(user_id)

    # Use Firestore transaction for ATOMIC read-check-write
    # This prevents race conditions where parallel requests see same credit balance
    @firestore.transactional
    def deduct_in_transaction(transaction, user_ref, credit_cost, tool_id):
        # Read current credits within transaction
        user_doc = user_ref.get(transaction=transaction)

        if not user_doc.exists:
            raise ValueError("USER_NOT_FOUND")

        current_credits = user_doc.to_dict().get('credits', 0)

        if current_credits < credit_cost:
            raise ValueError(f"INSUFFICIENT_CREDITS:{current_credits}:{credit_cost}")

        # Calculate new balance
        new_credits = current_credits - credit_cost

        # Update credits within transaction (atomic)
        transaction.update(user_ref, {
            'credits': new_credits,
            'updated_at': firestore.SERVER_TIMESTAMP,
        })

        return current_credits, new_credits

    try:
        # Execute transaction
        transaction = db.transaction()
        current_credits, new_credits = deduct_in_transaction(transaction, user_ref, credit_cost, tool_id)

        # Log to credit_history (outside transaction, not critical)
        user_ref.collection('credit_history').add({
            'amount': -credit_cost,
            'reason': f'generation_{tool_id}',
            'balance_after': new_credits,
            'type': 'deduct',
            'created_at': firestore.SERVER_TIMESTAMP,
        })

        logger.info(f"💰 [ATOMIC] Deducted {credit_cost} credits from {user_id} for {tool_id}. {current_credits} -> {new_credits}")
        return True

    except ValueError as e:
        error_msg = str(e)
        if error_msg == "USER_NOT_FOUND":
            logger.warning(f"🚫 User not found: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        elif error_msg.startswith("INSUFFICIENT_CREDITS"):
            parts = error_msg.split(":")
            current = parts[1] if len(parts) > 1 else "?"
            needed = parts[2] if len(parts) > 2 else credit_cost
            logger.warning(f"🚫 Insufficient credits for {user_id}: has {current}, needs {needed}")
            raise HTTPException(
                status_code=402,  # Payment Required
                detail=f"Insufficient credits. Have: {current}, Need: {needed}"
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Transaction error for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Credit deduction failed")

# In-memory job storage (use Redis/DB in production)
jobs_store: Dict[str, Dict[str, Any]] = {}


# ==================== GENERIC ENDPOINT ====================

@router.post("/generate", response_model=JobResponse)
async def generate(
    request: GenericAIRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Universal generation endpoint for any AI tool.
    SECURED: Requires Firebase Auth token + sufficient credits.
    Credits are deducted BEFORE processing starts.
    """
    tool_id = request.tool_id

    # Validate tool exists
    if tool_id not in AI_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool: {tool_id}. Check /api/v1/tools for available tools."
        )

    # Get credit cost for this tool
    # Special handling for text_to_video: dynamic pricing based on duration & resolution
    if tool_id == "text_to_video":
        duration = request.duration or 5
        resolution = request.resolution or "720p"
        # Pricing per second: 480p=6, 720p=9, 1080p=15 credits
        credits_per_second = {"480p": 6, "720p": 9, "1080p": 15}.get(resolution, 9)
        credit_cost = duration * credits_per_second
        logger.info(f"💰 [TEXT_TO_VIDEO] Dynamic pricing: {duration}s @ {resolution} = {credit_cost} credits")
    else:
        credit_cost = replicate_service.get_credit_cost(tool_id)

    # CHECK AND DEDUCT CREDITS BEFORE PROCESSING
    # This prevents unauthorized generation
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    logger.info(f"🎨 Starting job for user {user_id}, tool {tool_id}, cost {credit_cost} credits")

    # Create job
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,  # Track which user created this job
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "input": request.model_dump(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }
    jobs_store[job_id] = job

    # Prepare input params - accepts both URLs and base64 data URIs
    input_params = {
        "image_url": request.image_url,  # Can be URL or data:image/...;base64,...
        "image_url_2": request.image_url_2,
        "mask_url": request.mask_url,  # Mask for inpainting tools
        "audio_url": request.audio_url,  # Audio for talking_head
        "video_url": request.video_url,  # Video for video_expand
        "style_url": request.style_url,  # Style image for style_transfer
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "style": request.style,
        "aspect_ratio": request.aspect_ratio,
        "quality": request.quality,
        "duration": request.duration,
        "resolution": request.resolution,  # 480p, 720p, 1080p for video
        **(request.params or {}),
    }

    # Start background processing
    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message=f"Job started for {tool_id}",
        job_id=job_id,
        tool_id=tool_id,
        status=JobStatus.PENDING,
        estimated_time=replicate_service.get_estimated_time(tool_id),
        credits_used=credit_cost,
    )


async def process_job(job_id: str, tool_id: str, input_params: Dict[str, Any]):
    """Background task to process AI job."""
    try:
        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        logger.info(f"Processing job {job_id} for tool {tool_id}")

        success, result, client_processing = await replicate_service.run_prediction(tool_id, input_params)

        if success:
            jobs_store[job_id]["status"] = JobStatus.COMPLETED

            # Handle client-side processing (e.g., portrait_mode blur_background)
            if client_processing == "blur_background":
                # Store both the cutout URL and original URL for client-side composite
                if isinstance(result, dict):
                    result["original_url"] = input_params.get("image_url")
                    result["blur_background"] = True
                else:
                    # Result is just a URL string, convert to dict
                    result = {
                        "url": result,
                        "original_url": input_params.get("image_url"),
                        "blur_background": True
                    }

            jobs_store[job_id]["result"] = result
            jobs_store[job_id]["completed_at"] = datetime.utcnow()
            logger.info(f"Job {job_id} completed successfully")
        else:
            jobs_store[job_id]["status"] = JobStatus.FAILED
            jobs_store[job_id]["error"] = result
            logger.error(f"Job {job_id} failed: {result}")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        jobs_store[job_id]["status"] = JobStatus.FAILED
        jobs_store[job_id]["error"] = str(e)


@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Get the status of a generation job.
    SECURED: Only the job owner can check status.
    """
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]

    # Security: Verify job ownership
    job_owner = job.get("user_id")
    if job_owner and job_owner != user_id:
        logger.warning(f"🚫 User {user_id} tried to access job {job_id} owned by {job_owner}")
        raise HTTPException(status_code=403, detail="Access denied: not your job")

    result_url = None
    if job["result"] and isinstance(job["result"], dict):
        # Try to get single url first
        result_url = job["result"].get("url")
        # If not found, try urls list
        if not result_url:
            urls = job["result"].get("urls")
            if urls and isinstance(urls, list) and len(urls) > 0:
                result_url = urls[0]
        # If still not found, try result string
        if not result_url:
            result_str = job["result"].get("result")
            if result_str and isinstance(result_str, str) and result_str.startswith("http"):
                result_url = result_str

    # Build metadata with tool_id and any extra result data
    metadata = {"tool_id": job["tool_id"]}

    # Add extra fields from result for client-side processing
    if job["result"] and isinstance(job["result"], dict):
        # Video stitching
        if job["result"].get("stitch_on_client"):
            metadata["stitch_on_client"] = True
            metadata["original_url"] = job["result"].get("original_url")
            metadata["continuation_url"] = job["result"].get("continuation_url")

        # Portrait mode blur_background (bokeh effect)
        if job["result"].get("blur_background"):
            metadata["blur_background"] = True
            metadata["original_url"] = job["result"].get("original_url")

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=100.0 if job["status"] == JobStatus.COMPLETED else 0.0,
        result_url=result_url,
        error=job.get("error"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        metadata=metadata,
    )


# ==================== VIDEO AI ENDPOINTS ====================
# All endpoints below require Firebase Auth + Credit check

@router.post("/video/text-to-video", response_model=JobResponse)
async def text_to_video(
    request: TextToVideoRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Generate video from text prompt. SECURED."""
    tool_id = "text_to_video"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "duration": request.duration,
        "aspect_ratio": request.aspect_ratio.value,
        "style": request.style.value,
        "seed": request.seed,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Text-to-video generation started",
        job_id=job_id,
        tool_id=tool_id,
        status=JobStatus.PENDING,
        estimated_time=120,
        credits_used=credit_cost,
    )


@router.post("/video/image-to-video", response_model=JobResponse)
async def image_to_video(
    request: ImageToVideoRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Animate an image into video. SECURED."""
    tool_id = "image_to_video"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "prompt": request.prompt,
        "duration": request.duration,
        "motion_strength": request.motion_strength,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Image-to-video animation started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=90,
        credits_used=credit_cost,
    )


@router.post("/video/ai-hug", response_model=JobResponse)
async def ai_hug(
    request: AIHugRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Create hugging animation from two photos. SECURED."""
    tool_id = "ai_hug"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "person1_url": str(request.person1_image_url),
        "person2_url": str(request.person2_image_url),
        "hug_style": request.hug_style,
        "duration": request.duration,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="AI Hug generation started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=120,
        credits_used=credit_cost,
    )


@router.post("/video/face-swap", response_model=JobResponse)
async def face_swap_video(
    request: FaceSwapVideoRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Swap face in a video. SECURED."""
    tool_id = "face_swap_video"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "source_face_url": str(request.source_face_url),
        "target_video_url": str(request.target_video_url),
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Face swap video started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=90,
        credits_used=credit_cost,
    )


@router.post("/video/talking-avatar", response_model=JobResponse)
async def talking_avatar(
    request: TalkingAvatarRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Create talking avatar from photo. SECURED."""
    tool_id = "talking_avatar"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "audio_url": str(request.audio_url) if request.audio_url else None,
        "text": request.text,
        "expression": request.expression,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Talking avatar generation started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=60,
        credits_used=credit_cost,
    )


# ==================== PHOTO ENHANCE ENDPOINTS ====================
# All endpoints below require Firebase Auth + Credit check

@router.post("/enhance/upscale", response_model=JobResponse)
async def upscale_image(
    request: UpscaleRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Upscale image to 4K quality. SECURED."""
    tool_id = "4k_upscale"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "scale": request.scale,
        "face_enhance": request.face_enhance,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="4K upscale started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=30,
        credits_used=credit_cost,
    )


@router.post("/enhance/face-clarify", response_model=JobResponse)
async def face_clarify(
    request: FaceClarifyRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Enhance and clarify faces in image. SECURED."""
    tool_id = "face_clarify"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "version": request.version,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Face clarify started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=15,
        credits_used=credit_cost,
    )


@router.post("/enhance/denoise", response_model=JobResponse)
async def denoise_image(
    request: DenoiseRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Remove noise from image. SECURED."""
    tool_id = "denoise"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "noise_level": request.noise_level,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Denoise started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=20,
        credits_used=credit_cost,
    )


@router.post("/enhance/colorize", response_model=JobResponse)
async def colorize_image(
    request: ColorizeRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Colorize black & white photo. SECURED."""
    tool_id = "colorize"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "render_factor": request.render_factor,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Colorization started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=30,
        credits_used=credit_cost,
    )


@router.post("/enhance/restore", response_model=JobResponse)
async def restore_old_photo(
    request: OldPhotoRestoreRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Restore old damaged photo. SECURED."""
    tool_id = "old_photo_restore"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "with_scratch": request.with_scratch,
        "face_enhance": request.face_enhance,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Photo restoration started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=45,
        credits_used=credit_cost,
    )


# ==================== MAGIC EDIT ENDPOINTS ====================
# All endpoints below require Firebase Auth + Credit check

@router.post("/edit/remove-background", response_model=JobResponse)
async def remove_background(
    request: BackgroundRemoveRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Remove background from image. SECURED."""
    tool_id = "background_remove"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Background removal started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=10,
        credits_used=credit_cost,
    )


@router.post("/edit/change-background", response_model=JobResponse)
async def change_background(
    request: BackgroundChangeRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Change background of image. SECURED."""
    tool_id = "background_change"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "new_background_prompt": request.new_background_prompt,
        "mask_url": str(request.mask_url) if request.mask_url else None,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Background change started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=30,
        credits_used=credit_cost,
    )


@router.post("/edit/remove-object", response_model=JobResponse)
async def remove_object(
    request: ObjectRemoveRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Remove object from image. SECURED."""
    tool_id = "object_remove"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "mask_url": str(request.mask_url),
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Object removal started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=20,
        credits_used=credit_cost,
    )


@router.post("/edit/face-swap", response_model=JobResponse)
async def face_swap(
    request: FaceSwapRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Swap faces in photo. SECURED."""
    tool_id = "face_swap"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "source_face_url": str(request.source_face_url),
        "target_image_url": str(request.target_image_url),
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Face swap started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=20,
        credits_used=credit_cost,
    )


@router.post("/edit/sketch-to-image", response_model=JobResponse)
async def sketch_to_image(
    request: SketchToImageRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Convert sketch to realistic image. SECURED."""
    tool_id = "sketch_to_image"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "sketch_url": str(request.sketch_url),
        "prompt": request.prompt,
        "style": request.style.value,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Sketch to image started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=30,
        credits_used=credit_cost,
    )


@router.post("/edit/style-transfer", response_model=JobResponse)
async def style_transfer(
    request: StyleTransferRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Apply artistic style to image. SECURED."""
    tool_id = "style_transfer"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "content_url": str(request.content_url),
        "style": request.style,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Style transfer started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=25,
        credits_used=credit_cost,
    )


@router.post("/edit/age-transform", response_model=JobResponse)
async def age_transform(
    request: AgeTransformRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Transform age in photo. SECURED."""
    tool_id = "age_transform"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "target_age": request.target_age,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Age transform started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=25,
        credits_used=credit_cost,
    )


@router.post("/edit/expression", response_model=JobResponse)
async def edit_expression(
    request: ExpressionEditRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_firebase_token)
):
    """Edit facial expression. SECURED."""
    tool_id = "expression_edit"
    credit_cost = replicate_service.get_credit_cost(tool_id)
    await check_and_deduct_credits(user_id, tool_id, credit_cost)

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": tool_id,
        "user_id": user_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
        "credits_charged": credit_cost,
    }

    input_params = {
        "image_url": str(request.image_url),
        "expression": request.expression,
        "intensity": request.intensity,
    }

    background_tasks.add_task(process_job, job_id, tool_id, input_params)

    return JobResponse(
        success=True,
        message="Expression edit started",
        job_id=job_id,
        tool_id=tool_id,
        estimated_time=20,
        credits_used=credit_cost,
    )


# ==================== REVIEWER MODE ====================
# Secure endpoint for Google Play reviewers only (Android)

from pydantic import BaseModel

class ReviewerModeRequest(BaseModel):
    secret_code: str

class ReviewerModeResponse(BaseModel):
    success: bool
    message: str
    credits_added: int = 0

@router.post("/activate-reviewer-mode", response_model=ReviewerModeResponse)
async def activate_reviewer_mode(
    request: ReviewerModeRequest,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Activate reviewer mode for Google Play reviewers.
    SECURITY:
    - Secret code verified against Firestore (not hardcoded)
    - Each user can only activate ONCE (prevents abuse)
    - Remote flag must be enabled in Firestore
    - Adds real credits to Firestore (not just local)
    """
    db = get_firestore_client()

    # 1. Check remote flag and get secret code from Firestore
    try:
        settings_doc = db.collection('app_settings').document('reviewer_mode').get()

        if not settings_doc.exists:
            logger.warning(f"🚫 Reviewer mode settings not found")
            raise HTTPException(status_code=403, detail="Reviewer mode not available")

        settings = settings_doc.to_dict()
        is_enabled = settings.get('enabled', False)
        remote_secret = settings.get('secret_code', '')
        credits_to_add = settings.get('credits_amount', 1000)  # Default 1000

        if not is_enabled:
            logger.warning(f"🚫 Reviewer mode disabled remotely")
            raise HTTPException(status_code=403, detail="Reviewer mode not available")

        if not remote_secret:
            logger.warning(f"🚫 No secret code configured")
            raise HTTPException(status_code=403, detail="Reviewer mode not configured")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching reviewer settings: {e}")
        raise HTTPException(status_code=500, detail="Configuration error")

    # 2. Verify secret code
    if request.secret_code.strip() != remote_secret:
        logger.warning(f"🚫 Invalid reviewer code attempt by user {user_id}")
        raise HTTPException(status_code=403, detail="Invalid code")

    # 3. Check if user already activated (prevent reuse)
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()

    if not user_doc.exists:
        logger.warning(f"🚫 User not found: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_doc.to_dict()
    if user_data.get('reviewer_mode_activated', False):
        logger.warning(f"🚫 User {user_id} already activated reviewer mode")
        raise HTTPException(status_code=400, detail="Already activated")

    # 4. Add credits and mark as reviewer (atomic transaction)
    try:
        @firestore.transactional
        def activate_reviewer(transaction, user_ref, credits_to_add):
            user_doc = user_ref.get(transaction=transaction)
            current_credits = user_doc.to_dict().get('credits', 0)
            new_credits = current_credits + credits_to_add

            transaction.update(user_ref, {
                'credits': new_credits,
                'reviewer_mode_activated': True,
                'is_reviewer_premium': True,  # Grants premium access (isSubscribed = true)
                'reviewer_activated_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
            })

            return new_credits

        transaction = db.transaction()
        new_credits = activate_reviewer(transaction, user_ref, credits_to_add)

        # Log to credit history
        user_ref.collection('credit_history').add({
            'amount': credits_to_add,
            'reason': 'reviewer_mode_activation',
            'balance_after': new_credits,
            'type': 'add',
            'created_at': firestore.SERVER_TIMESTAMP,
        })

        logger.info(f"✅ Reviewer mode activated for user {user_id}, added {credits_to_add} credits")

        return ReviewerModeResponse(
            success=True,
            message="Reviewer mode activated successfully",
            credits_added=credits_to_add,
        )

    except Exception as e:
        logger.error(f"❌ Error activating reviewer mode: {e}")
        raise HTTPException(status_code=500, detail="Activation failed")
