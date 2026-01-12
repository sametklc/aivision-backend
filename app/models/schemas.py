"""
AIVision API - Pydantic Schemas
Request/Response models for all endpoints
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Literal
from enum import Enum
from datetime import datetime


# ==================== ENUMS ====================

class ToolCategory(str, Enum):
    VIDEO_AI = "VIDEO_AI"
    PHOTO_ENHANCE = "PHOTO_ENHANCE"
    MAGIC_EDIT = "MAGIC_EDIT"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AspectRatio(str, Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"


class StyleType(str, Enum):
    REALISTIC = "realistic"
    CINEMATIC = "cinematic"
    ANIME = "anime"
    ARTISTIC = "artistic"
    VINTAGE = "vintage"
    NEON = "neon"


class QualityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


# ==================== BASE SCHEMAS ====================

class BaseRequest(BaseModel):
    """Base request with common fields"""
    tool_id: str = Field(..., description="Tool identifier")
    user_id: Optional[str] = Field(None, description="User ID for credit tracking")


class BaseResponse(BaseModel):
    """Base response with common fields"""
    success: bool
    message: str
    job_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    error_code: str
    details: Optional[dict] = None


# ==================== VIDEO AI SCHEMAS ====================

class TextToVideoRequest(BaseRequest):
    """Text to Video generation request"""
    tool_id: str = "text_to_video"
    prompt: str = Field(..., min_length=10, max_length=2000, description="Video description prompt")
    negative_prompt: Optional[str] = Field(None, description="What to avoid in video")
    duration: int = Field(4, ge=2, le=10, description="Video duration in seconds")
    aspect_ratio: AspectRatio = Field(AspectRatio.RATIO_16_9, description="Video aspect ratio")
    style: StyleType = Field(StyleType.CINEMATIC, description="Visual style")
    quality: QualityLevel = Field(QualityLevel.HIGH, description="Output quality")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ImageToVideoRequest(BaseRequest):
    """Image to Video animation request"""
    tool_id: str = "image_to_video"
    image_url: HttpUrl = Field(..., description="Source image URL")
    prompt: Optional[str] = Field(None, description="Motion/animation prompt")
    duration: int = Field(4, ge=2, le=10, description="Video duration in seconds")
    motion_strength: float = Field(0.7, ge=0.1, le=1.0, description="Amount of motion")


class AIHugRequest(BaseRequest):
    """AI Hug generation - two people hugging animation"""
    tool_id: str = "ai_hug"
    person1_image_url: HttpUrl = Field(..., description="First person image URL")
    person2_image_url: HttpUrl = Field(..., description="Second person image URL")
    hug_style: Literal["friendly", "romantic", "casual"] = Field("friendly")
    duration: int = Field(4, ge=2, le=6)


class FaceSwapVideoRequest(BaseRequest):
    """Face swap in video request"""
    tool_id: str = "face_swap_video"
    source_face_url: HttpUrl = Field(..., description="Source face image URL")
    target_video_url: HttpUrl = Field(..., description="Target video URL")


class TalkingAvatarRequest(BaseRequest):
    """Create talking avatar from photo"""
    tool_id: str = "talking_avatar"
    image_url: HttpUrl = Field(..., description="Portrait image URL")
    audio_url: Optional[HttpUrl] = Field(None, description="Audio URL for lip sync")
    text: Optional[str] = Field(None, description="Text for TTS if no audio provided")
    expression: Literal["neutral", "happy", "sad", "surprised"] = Field("neutral")


# ==================== PHOTO ENHANCE SCHEMAS ====================

class UpscaleRequest(BaseRequest):
    """4K upscale request"""
    tool_id: str = "4k_upscale"
    image_url: HttpUrl = Field(..., description="Image URL to upscale")
    scale: int = Field(4, ge=2, le=8, description="Upscale factor")
    face_enhance: bool = Field(True, description="Apply GFPGAN face enhancement")


class FaceClarifyRequest(BaseRequest):
    """Face clarification request"""
    tool_id: str = "face_clarify"
    image_url: HttpUrl = Field(..., description="Image URL with faces to clarify")
    version: Literal["1.3", "1.4"] = Field("1.4", description="GFPGAN version")


class DenoiseRequest(BaseRequest):
    """Image denoising request"""
    tool_id: str = "denoise"
    image_url: HttpUrl = Field(..., description="Noisy image URL")
    noise_level: int = Field(25, ge=5, le=50, description="Estimated noise level")


class ColorizeRequest(BaseRequest):
    """Photo colorization request"""
    tool_id: str = "colorize"
    image_url: HttpUrl = Field(..., description="Black & white image URL")
    render_factor: int = Field(35, ge=10, le=50, description="Colorization quality")


class OldPhotoRestoreRequest(BaseRequest):
    """Old photo restoration request"""
    tool_id: str = "old_photo_restore"
    image_url: HttpUrl = Field(..., description="Old photo URL")
    with_scratch: bool = Field(True, description="Remove scratches")
    face_enhance: bool = Field(True, description="Enhance faces")


# ==================== MAGIC EDIT SCHEMAS ====================

class BackgroundRemoveRequest(BaseRequest):
    """Background removal request"""
    tool_id: str = "background_remove"
    image_url: HttpUrl = Field(..., description="Image URL")


class BackgroundChangeRequest(BaseRequest):
    """Background change request"""
    tool_id: str = "background_change"
    image_url: HttpUrl = Field(..., description="Image URL")
    new_background_prompt: str = Field(..., description="New background description")
    mask_url: Optional[HttpUrl] = Field(None, description="Custom mask URL")


class ObjectRemoveRequest(BaseRequest):
    """Object removal request"""
    tool_id: str = "object_remove"
    image_url: HttpUrl = Field(..., description="Image URL")
    mask_url: HttpUrl = Field(..., description="Mask indicating object to remove")


class FaceSwapRequest(BaseRequest):
    """Face swap in photos request"""
    tool_id: str = "face_swap"
    source_face_url: HttpUrl = Field(..., description="Source face image")
    target_image_url: HttpUrl = Field(..., description="Target image")


class SketchToImageRequest(BaseRequest):
    """Sketch to image request"""
    tool_id: str = "sketch_to_image"
    sketch_url: HttpUrl = Field(..., description="Sketch image URL")
    prompt: str = Field(..., description="Description of desired output")
    style: StyleType = Field(StyleType.REALISTIC)


class StyleTransferRequest(BaseRequest):
    """Style transfer request"""
    tool_id: str = "style_transfer"
    content_url: HttpUrl = Field(..., description="Content image URL")
    style: Literal["ghibli", "vangogh", "picasso", "monet", "comic", "anime"] = Field("ghibli")


class AgeTransformRequest(BaseRequest):
    """Age transformation request"""
    tool_id: str = "age_transform"
    image_url: HttpUrl = Field(..., description="Face image URL")
    target_age: Literal["baby", "child", "teen", "young", "middle", "old"] = Field("old")


class ExpressionEditRequest(BaseRequest):
    """Expression editing request"""
    tool_id: str = "expression_edit"
    image_url: HttpUrl = Field(..., description="Face image URL")
    expression: Literal["smile", "sad", "angry", "surprised", "neutral"] = Field("smile")
    intensity: float = Field(0.7, ge=0.1, le=1.0)


# ==================== GENERIC AI REQUEST ====================

class GenericAIRequest(BaseRequest):
    """Generic AI request for any tool"""
    # Accept both URLs and base64 data URIs
    image_url: Optional[str] = Field(None, description="Primary image URL or base64 data URI")
    image_url_2: Optional[str] = Field(None, description="Secondary image URL or base64 data URI")
    mask_url: Optional[str] = Field(None, description="Mask image URL or base64 data URI for inpainting")
    audio_url: Optional[str] = Field(None, description="Audio URL or base64 data URI for talking_head")
    video_url: Optional[str] = Field(None, description="Video URL for video_expand")
    prompt: Optional[str] = Field(None, description="Text prompt")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    style: Optional[str] = Field(None, description="Style selection")
    aspect_ratio: Optional[str] = Field(None, description="Aspect ratio")
    quality: Optional[str] = Field("high", description="Quality level")
    duration: Optional[int] = Field(5, description="Duration for video tools (2-12 seconds)")
    resolution: Optional[str] = Field("720p", description="Video resolution: 480p, 720p, 1080p")

    # Additional parameters as dict for tool-specific options
    params: Optional[dict] = Field(default_factory=dict, description="Tool-specific parameters")


# ==================== JOB RESPONSE SCHEMAS ====================

class JobResponse(BaseResponse):
    """Job creation response"""
    job_id: str
    tool_id: str
    status: JobStatus = JobStatus.PENDING
    estimated_time: Optional[int] = Field(None, description="Estimated completion time in seconds")
    credits_used: int = 0


class JobStatusResponse(BaseModel):
    """Job status check response"""
    job_id: str
    status: JobStatus
    progress: float = Field(0, ge=0, le=100, description="Progress percentage")
    result_url: Optional[str] = None  # Can be URL or data URI
    thumbnail_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    metadata: Optional[dict] = None


# ==================== USER & CREDITS SCHEMAS ====================

class UserCredits(BaseModel):
    """User credits information"""
    user_id: str
    credits_balance: int
    credits_used_today: int
    subscription_tier: Literal["free", "pro", "unlimited"] = "free"


class CreditTransaction(BaseModel):
    """Credit transaction record"""
    transaction_id: str
    user_id: str
    amount: int
    transaction_type: Literal["use", "purchase", "bonus", "refund"]
    tool_id: Optional[str] = None
    created_at: datetime


# ==================== TOOL INFO SCHEMAS ====================

class ToolInfo(BaseModel):
    """Tool information schema"""
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    credit_cost: int
    input_type: Literal["text_only", "image_text", "image_only", "dual_image"]
    supports_styles: bool = False
    supports_aspect_ratio: bool = False
    max_duration: Optional[int] = None
    example_prompt: Optional[str] = None


class ToolsListResponse(BaseModel):
    """Response containing list of available tools"""
    success: bool = True
    tools: List[ToolInfo]
    total_count: int
