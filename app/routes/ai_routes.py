"""
AIVision API - AI Routes
Endpoints for all AI tools
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any
from loguru import logger
import uuid
from datetime import datetime

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

# In-memory job storage (use Redis/DB in production)
jobs_store: Dict[str, Dict[str, Any]] = {}


# ==================== GENERIC ENDPOINT ====================

@router.post("/generate", response_model=JobResponse)
async def generate(
    request: GenericAIRequest,
    background_tasks: BackgroundTasks
):
    """
    Universal generation endpoint for any AI tool.
    Accepts tool_id and routes to appropriate handler.
    """
    tool_id = request.tool_id

    # Validate tool exists
    if tool_id not in AI_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool: {tool_id}. Check /api/v1/tools for available tools."
        )

    # Create job
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "tool_id": tool_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "input": request.model_dump(),
        "result": None,
        "error": None,
    }
    jobs_store[job_id] = job

    # Prepare input params - accepts both URLs and base64 data URIs
    input_params = {
        "image_url": request.image_url,  # Can be URL or data:image/...;base64,...
        "image_url_2": request.image_url_2,
        "mask_url": request.mask_url,  # Mask for inpainting tools
        "audio_url": request.audio_url,  # Audio for talking_head
        "video_url": request.video_url,  # Video for video_expand
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
        credits_used=replicate_service.get_credit_cost(tool_id),
    )


async def process_job(job_id: str, tool_id: str, input_params: Dict[str, Any]):
    """Background task to process AI job."""
    try:
        jobs_store[job_id]["status"] = JobStatus.PROCESSING
        logger.info(f"Processing job {job_id} for tool {tool_id}")

        success, result, _ = await replicate_service.run_prediction(tool_id, input_params)

        if success:
            jobs_store[job_id]["status"] = JobStatus.COMPLETED
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
async def get_job_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_store[job_id]

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

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=100.0 if job["status"] == JobStatus.COMPLETED else 0.0,
        result_url=result_url,
        error=job.get("error"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        metadata={"tool_id": job["tool_id"]},
    )


# ==================== VIDEO AI ENDPOINTS ====================

@router.post("/video/text-to-video", response_model=JobResponse)
async def text_to_video(request: TextToVideoRequest, background_tasks: BackgroundTasks):
    """Generate video from text prompt."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "text_to_video",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "duration": request.duration,
        "aspect_ratio": request.aspect_ratio.value,
        "style": request.style.value,
        "seed": request.seed,
    }

    background_tasks.add_task(process_job, job_id, "text_to_video", input_params)

    return JobResponse(
        success=True,
        message="Text-to-video generation started",
        job_id=job_id,
        tool_id="text_to_video",
        status=JobStatus.PENDING,
        estimated_time=120,
        credits_used=15,
    )


@router.post("/video/image-to-video", response_model=JobResponse)
async def image_to_video(request: ImageToVideoRequest, background_tasks: BackgroundTasks):
    """Animate an image into video."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "image_to_video",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "prompt": request.prompt,
        "duration": request.duration,
        "motion_strength": request.motion_strength,
    }

    background_tasks.add_task(process_job, job_id, "image_to_video", input_params)

    return JobResponse(
        success=True,
        message="Image-to-video animation started",
        job_id=job_id,
        tool_id="image_to_video",
        estimated_time=90,
        credits_used=10,
    )


@router.post("/video/ai-hug", response_model=JobResponse)
async def ai_hug(request: AIHugRequest, background_tasks: BackgroundTasks):
    """Create hugging animation from two photos."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "ai_hug",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    # AI Hug requires preprocessing to combine two faces
    input_params = {
        "person1_url": str(request.person1_image_url),
        "person2_url": str(request.person2_image_url),
        "hug_style": request.hug_style,
        "duration": request.duration,
    }

    background_tasks.add_task(process_job, job_id, "ai_hug", input_params)

    return JobResponse(
        success=True,
        message="AI Hug generation started",
        job_id=job_id,
        tool_id="ai_hug",
        estimated_time=120,
        credits_used=12,
    )


@router.post("/video/face-swap", response_model=JobResponse)
async def face_swap_video(request: FaceSwapVideoRequest, background_tasks: BackgroundTasks):
    """Swap face in a video."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "face_swap_video",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "source_face_url": str(request.source_face_url),
        "target_video_url": str(request.target_video_url),
    }

    background_tasks.add_task(process_job, job_id, "face_swap_video", input_params)

    return JobResponse(
        success=True,
        message="Face swap video started",
        job_id=job_id,
        tool_id="face_swap_video",
        estimated_time=90,
        credits_used=10,
    )


@router.post("/video/talking-avatar", response_model=JobResponse)
async def talking_avatar(request: TalkingAvatarRequest, background_tasks: BackgroundTasks):
    """Create talking avatar from photo."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "talking_avatar",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "audio_url": str(request.audio_url) if request.audio_url else None,
        "text": request.text,
        "expression": request.expression,
    }

    background_tasks.add_task(process_job, job_id, "talking_avatar", input_params)

    return JobResponse(
        success=True,
        message="Talking avatar generation started",
        job_id=job_id,
        tool_id="talking_avatar",
        estimated_time=60,
        credits_used=8,
    )


# ==================== PHOTO ENHANCE ENDPOINTS ====================

@router.post("/enhance/upscale", response_model=JobResponse)
async def upscale_image(request: UpscaleRequest, background_tasks: BackgroundTasks):
    """Upscale image to 4K quality."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "4k_upscale",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "scale": request.scale,
        "face_enhance": request.face_enhance,
    }

    background_tasks.add_task(process_job, job_id, "4k_upscale", input_params)

    return JobResponse(
        success=True,
        message="4K upscale started",
        job_id=job_id,
        tool_id="4k_upscale",
        estimated_time=30,
        credits_used=3,
    )


@router.post("/enhance/face-clarify", response_model=JobResponse)
async def face_clarify(request: FaceClarifyRequest, background_tasks: BackgroundTasks):
    """Enhance and clarify faces in image."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "face_clarify",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "version": request.version,
    }

    background_tasks.add_task(process_job, job_id, "face_clarify", input_params)

    return JobResponse(
        success=True,
        message="Face clarify started",
        job_id=job_id,
        tool_id="face_clarify",
        estimated_time=15,
        credits_used=2,
    )


@router.post("/enhance/denoise", response_model=JobResponse)
async def denoise_image(request: DenoiseRequest, background_tasks: BackgroundTasks):
    """Remove noise from image."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "denoise",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "noise_level": request.noise_level,
    }

    background_tasks.add_task(process_job, job_id, "denoise", input_params)

    return JobResponse(
        success=True,
        message="Denoise started",
        job_id=job_id,
        tool_id="denoise",
        estimated_time=20,
        credits_used=2,
    )


@router.post("/enhance/colorize", response_model=JobResponse)
async def colorize_image(request: ColorizeRequest, background_tasks: BackgroundTasks):
    """Colorize black & white photo."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "colorize",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "render_factor": request.render_factor,
    }

    background_tasks.add_task(process_job, job_id, "colorize", input_params)

    return JobResponse(
        success=True,
        message="Colorization started",
        job_id=job_id,
        tool_id="colorize",
        estimated_time=30,
        credits_used=3,
    )


@router.post("/enhance/restore", response_model=JobResponse)
async def restore_old_photo(request: OldPhotoRestoreRequest, background_tasks: BackgroundTasks):
    """Restore old damaged photo."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "old_photo_restore",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "with_scratch": request.with_scratch,
        "face_enhance": request.face_enhance,
    }

    background_tasks.add_task(process_job, job_id, "old_photo_restore", input_params)

    return JobResponse(
        success=True,
        message="Photo restoration started",
        job_id=job_id,
        tool_id="old_photo_restore",
        estimated_time=45,
        credits_used=4,
    )


# ==================== MAGIC EDIT ENDPOINTS ====================

@router.post("/edit/remove-background", response_model=JobResponse)
async def remove_background(request: BackgroundRemoveRequest, background_tasks: BackgroundTasks):
    """Remove background from image."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "background_remove",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
    }

    background_tasks.add_task(process_job, job_id, "background_remove", input_params)

    return JobResponse(
        success=True,
        message="Background removal started",
        job_id=job_id,
        tool_id="background_remove",
        estimated_time=10,
        credits_used=2,
    )


@router.post("/edit/change-background", response_model=JobResponse)
async def change_background(request: BackgroundChangeRequest, background_tasks: BackgroundTasks):
    """Change background of image."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "background_change",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "new_background_prompt": request.new_background_prompt,
        "mask_url": str(request.mask_url) if request.mask_url else None,
    }

    background_tasks.add_task(process_job, job_id, "background_change", input_params)

    return JobResponse(
        success=True,
        message="Background change started",
        job_id=job_id,
        tool_id="background_change",
        estimated_time=30,
        credits_used=4,
    )


@router.post("/edit/remove-object", response_model=JobResponse)
async def remove_object(request: ObjectRemoveRequest, background_tasks: BackgroundTasks):
    """Remove object from image."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "object_remove",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "mask_url": str(request.mask_url),
    }

    background_tasks.add_task(process_job, job_id, "object_remove", input_params)

    return JobResponse(
        success=True,
        message="Object removal started",
        job_id=job_id,
        tool_id="object_remove",
        estimated_time=20,
        credits_used=3,
    )


@router.post("/edit/face-swap", response_model=JobResponse)
async def face_swap(request: FaceSwapRequest, background_tasks: BackgroundTasks):
    """Swap faces in photo."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "face_swap",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "source_face_url": str(request.source_face_url),
        "target_image_url": str(request.target_image_url),
    }

    background_tasks.add_task(process_job, job_id, "face_swap", input_params)

    return JobResponse(
        success=True,
        message="Face swap started",
        job_id=job_id,
        tool_id="face_swap",
        estimated_time=20,
        credits_used=3,
    )


@router.post("/edit/sketch-to-image", response_model=JobResponse)
async def sketch_to_image(request: SketchToImageRequest, background_tasks: BackgroundTasks):
    """Convert sketch to realistic image."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "sketch_to_image",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "sketch_url": str(request.sketch_url),
        "prompt": request.prompt,
        "style": request.style.value,
    }

    background_tasks.add_task(process_job, job_id, "sketch_to_image", input_params)

    return JobResponse(
        success=True,
        message="Sketch to image started",
        job_id=job_id,
        tool_id="sketch_to_image",
        estimated_time=30,
        credits_used=4,
    )


@router.post("/edit/style-transfer", response_model=JobResponse)
async def style_transfer(request: StyleTransferRequest, background_tasks: BackgroundTasks):
    """Apply artistic style to image."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "style_transfer",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "content_url": str(request.content_url),
        "style": request.style,
    }

    background_tasks.add_task(process_job, job_id, "style_transfer", input_params)

    return JobResponse(
        success=True,
        message="Style transfer started",
        job_id=job_id,
        tool_id="style_transfer",
        estimated_time=25,
        credits_used=3,
    )


@router.post("/edit/age-transform", response_model=JobResponse)
async def age_transform(request: AgeTransformRequest, background_tasks: BackgroundTasks):
    """Transform age in photo."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "age_transform",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "target_age": request.target_age,
    }

    background_tasks.add_task(process_job, job_id, "age_transform", input_params)

    return JobResponse(
        success=True,
        message="Age transform started",
        job_id=job_id,
        tool_id="age_transform",
        estimated_time=25,
        credits_used=3,
    )


@router.post("/edit/expression", response_model=JobResponse)
async def edit_expression(request: ExpressionEditRequest, background_tasks: BackgroundTasks):
    """Edit facial expression."""
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "job_id": job_id,
        "tool_id": "expression_edit",
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow(),
        "result": None,
        "error": None,
    }

    input_params = {
        "image_url": str(request.image_url),
        "expression": request.expression,
        "intensity": request.intensity,
    }

    background_tasks.add_task(process_job, job_id, "expression_edit", input_params)

    return JobResponse(
        success=True,
        message="Expression edit started",
        job_id=job_id,
        tool_id="expression_edit",
        estimated_time=20,
        credits_used=3,
    )
