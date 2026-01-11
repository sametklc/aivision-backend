"""
AIVision Backend Configuration
Environment variables and settings
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # App Config
    APP_NAME: str = "AIVision API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Replicate API (Primary AI Provider)
    REPLICATE_API_TOKEN: str = ""

    # Cloudinary (Image Storage)
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./aivision.db"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds

    # AI Model Defaults
    DEFAULT_VIDEO_DURATION: int = 4  # seconds
    DEFAULT_IMAGE_QUALITY: str = "high"
    DEFAULT_UPSCALE_FACTOR: int = 4

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# ══════════════════════════════════════════════════════════════════════════════
# COST CONTROL SETTINGS - HARDCODED LIMITS TO SAVE MONEY
# ══════════════════════════════════════════════════════════════════════════════
COST_LIMITS = {
    # Video Generation Limits
    "video": {
        "num_frames": 81,            # Minimum for wan-video
        "resolution": "480p",        # Lowest resolution
        "frames_per_second": 16,     # Default fps
    },

    # Magic Edit & Image Gen Limits
    "magic_edit": {
        "num_inference_steps": 4,    # Flux schnell uses 4 max
        "guidance_scale": 7.5,       # For SD inpainting
        "output_quality": 80,
        "megapixels": "0.25",        # Low resolution for flux
    },

    # Photo Enhancement Limits
    "photo_enhance": {
        "scale": 2,                  # Don't upscale by 4x or 8x
        "output_quality": 80,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# AI MODELS - VERIFIED REPLICATE MODELS WITH EXACT HASHES (January 2025)
# ══════════════════════════════════════════════════════════════════════════════
AI_MODELS = {
    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO AI (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════
    "ai_hug": {
        "model": "wan-video/wan-2.2-i2v-fast",
        "version": None,
        "category": "video",
        "description": "Create heartwarming hug animations",
        "cost_per_run": 0.05,
        "supports": ["image", "prompt"],
        "default_prompt": "two people hugging warmly, emotional moment, smooth motion",
    },
    "image_to_video": {
        "model": "wan-video/wan-2.2-i2v-fast",
        "version": None,
        "category": "video",
        "description": "Animate your images with AI",
        "cost_per_run": 0.05,
        "supports": ["image", "prompt", "duration"],
    },
    "text_to_video": {
        "model": "bytedance/seedance-1-lite",
        "version": None,
        "category": "video",
        "description": "Generate videos from text prompts",
        "cost_per_run": 0.25,
        "supports": ["prompt", "duration", "resolution"],
        "default_duration": 5,  # 2-12 seconds
        "default_resolution": "720p",  # 480p, 720p, 1080p
        "default_aspect_ratio": "16:9",
    },
    "talking_head": {
        "model": "lucataco/sadtalker",
        "version": "85c698db7c0a66d5011435d0191db323034e1da04b912a6d365833141b6a285b",
        "category": "video",
        "description": "Lip sync any portrait photo",
        "cost_per_run": 0.15,
        "supports": ["image", "audio"],
    },
    "video_expand": {
        "model": "wan-video/wan-2.2-i2v-fast",
        "version": None,
        "category": "video",
        "description": "Extend video canvas seamlessly",
        "cost_per_run": 0.05,
        "supports": ["video", "direction", "prompt"],
        "note": "Use last frame logic for continuation",
    },
    "style_transfer_video": {
        "model": "lucataco/animate-diff-vid2vid",
        "version": "e69bc3ee033ba546514eeccce95ec31964fa5834633d8d273138609aecae143c",
        "category": "video",
        "description": "Apply artistic styles to videos",
        "cost_per_run": 0.10,
        "supports": ["video", "style", "prompt"],
    },
    "super_slowmo": {
        "model": "google-research/frame-interpolation",
        "version": "4f88a16a13673a8b589c18866e540556170a5bcb2ccdc12de556e800e9456d3d",
        "category": "video",
        "description": "Create cinematic slow motion",
        "cost_per_run": 0.05,
        "supports": ["video"],
    },
    "video_upscale": {
        "model": "lucataco/real-esrgan-video",
        "version": "3e56ce4b57863bd03048b42bc09bdd4db20d427cca5fde9d8ae4dc60e1bb4775",
        "category": "video",
        "description": "Enhance video to 4K quality",
        "cost_per_run": 0.10,
        "supports": ["video", "scale"],
    },
    "video_bg_remove": {
        "model": "arielreplicate/robust_video_matting",
        "version": "73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
        "category": "video",
        "description": "Remove video background",
        "cost_per_run": 0.08,
        "supports": ["video"],
    },
    "face_swap_video": {
        "model": "yan-ops/face-swap",
        "version": "74e365021a8f6d744b7f8c0780211a729e8c895964f40f2f750b284852033096",
        "category": "video",
        "description": "Swap faces in videos",
        "cost_per_run": 0.12,
        "supports": ["video", "image"],
    },
    "script_to_video": {
        "model": "minimax/video-01",
        "version": None,
        "category": "video",
        "description": "Turn scripts into videos",
        "cost_per_run": 0.50,
        "supports": ["prompt"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PHOTO ENHANCE (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════
    "face_clarify": {
        "model": "tencentarc/gfpgan",
        "version": "0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
        "category": "photo_enhance",
        "description": "Restore face details in photos",
        "cost_per_run": 0.003,
        "supports": ["image"],
    },
    "old_photo_restore": {
        "model": "microsoft/bringing-old-photos-back-to-life",
        "version": "c75db81db6cbd809d93cc3b7e7a088a351a3349c9fa02b6d393e35e0d51ba799",
        "category": "photo_enhance",
        "description": "Bring old memories back to life",
        "cost_per_run": 0.005,
        "supports": ["image"],
    },
    "colorize": {
        "model": "arielreplicate/deoldify_image",
        "version": "0da600fab0c45a66211339f1c16b71345d22f26ef5fea3dca1bb90bb5711e950",
        "category": "photo_enhance",
        "description": "Add color to B&W photos",
        "cost_per_run": 0.005,
        "supports": ["image"],
    },
    "4k_upscale": {
        "model": "nightmareai/real-esrgan",
        "version": None,  # Use latest version
        "category": "photo_enhance",
        "description": "Enhance images to 4K resolution",
        "cost_per_run": 0.002,
        "supports": ["image", "scale"],
    },
    "relight": {
        "model": "zsxkib/ic-light",
        "version": "d41bcb10d8c159868f4cfbd7c6a2ca01484f7d39e4613419d5952c61562f1ba7",
        "category": "photo_enhance",
        "description": "Change lighting in photos",
        "cost_per_run": 0.008,
        "supports": ["image", "prompt"],
        "default_prompt": "professional studio lighting, soft shadows",
    },
    "low_light_fix": {
        "model": "megvii-research/nafnet",
        "version": "018241a6c880319404eaa2714b764313e27e11f950a7ff0a7b5b37b27b74dcf7",
        "category": "photo_enhance",
        "description": "Brighten dark photos naturally",
        "cost_per_run": 0.003,
        "supports": ["image"],
    },
    "scratch_remover": {
        "model": "microsoft/bringing-old-photos-back-to-life",
        "version": "c75db81db6cbd809d93cc3b7e7a088a351a3349c9fa02b6d393e35e0d51ba799",
        "category": "photo_enhance",
        "description": "Remove scratches and damage",
        "cost_per_run": 0.005,
        "supports": ["image"],
    },
    "denoise": {
        "model": "sczhou/codeformer",
        "version": "cc4956dd26fa5a7185d5660cc9100fab1b8070a1d1654a8bb5eb6d443b020bb2",
        "category": "photo_enhance",
        "description": "Remove noise and grain",
        "cost_per_run": 0.005,
        "supports": ["image"],
    },
    "anime_yourself": {
        "model": "aaronaftab/mirage-ghibli",
        "version": "166efd159b4138da932522bc5af40d39194033f587d9bdbab1e594119eae3e7f",
        "category": "photo_enhance",
        "description": "Transform into anime style",
        "cost_per_run": 0.008,
        "supports": ["image", "prompt"],
        "default_prompt": "anime style portrait, ghibli style, beautiful detailed anime art, vibrant colors",
    },
    "portrait_mode": {
        "model": "lucataco/remove-bg",
        "version": "95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1",
        "category": "photo_enhance",
        "description": "Add professional bokeh effect",
        "cost_per_run": 0.0004,
        "supports": ["image"],
        "client_side_processing": "blur_background",
    },
    "color_correct": {
        "model": "google-research/maxim",
        "version": None,  # Use latest version
        "category": "photo_enhance",
        "description": "Professional color grading",
        "cost_per_run": 0.005,
        "supports": ["image"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MAGIC EDIT (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════
    "magic_eraser": {
        "model": "bria/eraser",
        "version": "893e924eecc119a0c5fbfa5d98401118dcbf0662574eb8d2c01be5749756cbd4",
        "category": "magic_edit",
        "description": "Remove unwanted objects",
        "cost_per_run": 0.003,
        "supports": ["image", "mask"],
    },
    "ai_headshot": {
        "model": "tencentarc/photomaker",
        "version": "ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
        "category": "magic_edit",
        "description": "Professional headshots instantly",
        "cost_per_run": 0.015,
        "supports": ["image", "prompt"],
        "default_prompt": "professional headshot portrait, studio lighting, business photo, high quality, sharp focus",
    },
    "clothes_swap": {
        "model": "cuuupid/idm-vton",
        "version": "0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
        "category": "magic_edit",
        "description": "Try on different outfits",
        "cost_per_run": 0.015,
        "supports": ["image", "garment_image"],
    },
    "bg_remix": {
        "model": "black-forest-labs/flux-fill-schnell",
        "version": None,
        "category": "magic_edit",
        "description": "Change background creatively",
        "cost_per_run": 0.003,
        "supports": ["image", "mask", "prompt"],
    },
    "sticker_maker": {
        "model": "lucataco/remove-bg",
        "version": "95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1",
        "category": "magic_edit",
        "description": "Create custom stickers",
        "cost_per_run": 0.0004,
        "supports": ["image"],
        "client_side_processing": "add_outline",
    },
    "outpainting": {
        "model": "black-forest-labs/flux-fill-schnell",
        "version": None,
        "category": "magic_edit",
        "description": "Expand image boundaries",
        "cost_per_run": 0.003,
        "supports": ["image", "mask", "prompt"],
    },
    "sky_replace": {
        "model": "black-forest-labs/flux-fill-schnell",
        "version": None,
        "category": "magic_edit",
        "description": "Change sky in photos",
        "cost_per_run": 0.003,
        "supports": ["image", "mask", "prompt"],
        "default_prompt": "beautiful dramatic sky, sunset clouds, golden hour, vibrant colors",
    },
    "interior_design": {
        "model": "adirik/interior-design",
        "version": "76619655695840563d7ce461876a43876e5df48545a0b730598711e133dfd535",
        "category": "magic_edit",
        "description": "Redesign room interiors",
        "cost_per_run": 0.012,
        "supports": ["image", "prompt"],
        "default_prompt": "modern interior design, stylish furniture, professional photography",
    },
    "product_shoot": {
        "model": "zsxkib/ic-light",
        "version": "d41bcb10d8c159868f4cfbd7c6a2ca01484f7d39e4613419d5952c61562f1ba7",
        "category": "magic_edit",
        "description": "Professional product photos",
        "cost_per_run": 0.008,
        "supports": ["image", "prompt"],
        "default_prompt": "professional product photography, studio lighting, white background, commercial quality",
    },
    "text_effects": {
        "model": "jagilley/controlnet-canny",
        "version": "aff48af9c68d162388d230a2ab003f68d2638d88307bdaf1c2f1ac95079c9613",
        "category": "magic_edit",
        "description": "Add stylish text overlays",
        "cost_per_run": 0.005,
        "supports": ["image", "prompt"],
    },
    "tattoo_tryon": {
        "model": "black-forest-labs/flux-fill-schnell",
        "version": None,
        "category": "magic_edit",
        "description": "Preview tattoos on your body",
        "cost_per_run": 0.003,
        "supports": ["image", "mask", "prompt"],
    },
}

# Tool categories for frontend - All 33 Tools
TOOL_CATEGORIES = {
    "VIDEO_AI": [
        "ai_hug", "image_to_video", "text_to_video", "talking_head",
        "video_expand", "style_transfer_video", "super_slowmo",
        "video_upscale", "video_bg_remove", "face_swap_video", "script_to_video"
    ],
    "PHOTO_ENHANCE": [
        "face_clarify", "old_photo_restore", "colorize", "4k_upscale",
        "relight", "low_light_fix", "scratch_remover", "denoise",
        "anime_yourself", "portrait_mode", "color_correct"
    ],
    "MAGIC_EDIT": [
        "magic_eraser", "ai_headshot", "clothes_swap", "bg_remix",
        "sticker_maker", "outpainting", "sky_replace", "interior_design",
        "product_shoot", "text_effects", "tattoo_tryon"
    ]
}
