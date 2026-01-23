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

    # RevenueCat API (for purchase validation)
    REVENUECAT_API_KEY: str = ""  # Secret API key from RevenueCat dashboard
    REVENUECAT_PROJECT_ID: str = ""  # Your RevenueCat project ID

    # Internal API Key (for backend-only endpoints like refund)
    INTERNAL_API_KEY: str = ""  # Generate a strong random key

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds

    # CORS - Allowed origins (comma-separated in env, or empty for mobile-only)
    # For mobile apps, CORS doesn't apply, but this restricts web access
    ALLOWED_ORIGINS: str = ""  # e.g., "https://yourdomain.com,https://admin.yourdomain.com"

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
    "kling_video": {
        "model": "kwaivgi/kling-v1.6-pro",
        "version": None,  # Use latest
        "category": "video",
        "description": "High-quality video generation with Kling 1.6 Pro",
        "cost_per_run": 0.50,
        "supports": ["image", "prompt", "duration"],
        # Kling params
        "duration": 5,  # 5 or 10 seconds
        "cfg_scale": 0.5,  # 0-1, higher = more prompt adherence
        "aspect_ratio": "9:16",
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
        "description": "Make your photo talk with text or audio",
        "cost_per_run": 0.35,  # TTS + SadTalker combined
        "supports": ["image", "text", "audio"],
        "tts_model": "resemble-ai/chatterbox",
        "tts_version": "1b8422bc49635c20d0a84e387ed20879c0dd09254ecdb4e75dc4bec10ff94e97",
    },
    "video_expand": {
        "model": "kwaivgi/kling-v2.5-turbo-pro",
        "version": None,
        "category": "video",
        "description": "Extend video with AI imagination",
        "cost_per_run": 0.50,
        "supports": ["video", "prompt"],
        "note": "Uses Kling v2.5 for prompt-guided video continuation with new content generation",
        "default_prompt": "Continue the video smoothly with cinematic motion",
        "duration": 5,  # 5 or 10 seconds
    },
    "style_transfer_video": {
        "model": "lucataco/animate-diff-vid2vid",
        "version": "e69bc3ee033ba546514eeccce95ec31964fa5834633d8d273138609aecae143c",
        "category": "video",
        "description": "Apply artistic styles to videos",
        "cost_per_run": 0.10,
        "supports": ["video", "style", "prompt"],
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
        "model": "arabyai-replicate/roop_face_swap",
        "version": "11b6bf0f4e14d808f655e87e5448233cceff10a45f659d71539cafb7163b2e84",
        "category": "video",
        "description": "Swap faces in videos (Roop)",
        "cost_per_run": 0.15,
        "supports": ["video", "image"],
    },
    "script_to_video": {
        "model": "minimax/video-01",
        "version": "5aa835260ff7f40f4069c41185f72036accf99e29957bb4a3b3a911f3b6c1912",
        "category": "video",
        "description": "Turn scripts into videos with cinematic quality",
        "cost_per_run": 0.50,
        "supports": ["prompt", "aspect_ratio"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PHOTO ENHANCE (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════
    "face_clarify": {
        "model": "tencentarc/gfpgan",
        "version": "0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
        "category": "photo_enhance",
        "description": "Restore face details in photos",
        "cost_per_run": 0.10,
        "supports": ["image"],
    },
    "old_photo_restore": {
        "model": "microsoft/bringing-old-photos-back-to-life",
        "version": "c75db81db6cbd809d93cc3b7e7a088a351a3349c9fa02b6d393e35e0d51ba799",
        "category": "photo_enhance",
        "description": "Bring old memories back to life",
        "cost_per_run": 0.10,
        "supports": ["image"],
    },
    "colorize": {
        "model": "arielreplicate/deoldify_image",
        "version": "0da600fab0c45a66211339f1c16b71345d22f26ef5fea3dca1bb90bb5711e950",
        "category": "photo_enhance",
        "description": "Add color to B&W photos",
        "cost_per_run": 0.10,
        "supports": ["image"],
    },
    "4k_upscale": {
        "model": "cjwbw/supir-v0q",
        "version": "ede69f6a5ae7d09f769d683347325b08d2f83a93d136ed89747941205e0a71da",
        "category": "photo_enhance",
        "description": "Enhance images to 4K resolution",
        "cost_per_run": 0.10,
        "supports": ["image", "scale"],
    },
    "relight": {
        "model": "zsxkib/ic-light",
        "version": "d41bcb10d8c159868f4cfbd7c6a2ca01484f7d39e4613419d5952c61562f1ba7",
        "category": "photo_enhance",
        "description": "Change lighting in photos",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
        "default_prompt": "soft natural lighting, professional photo",
    },
    "low_light_fix": {
        "model": "megvii-research/nafnet",
        "version": "018241a6c880319404eaa2714b764313e27e11f950a7ff0a7b5b37b27b74dcf7",
        "category": "photo_enhance",
        "description": "Brighten dark photos naturally",
        "cost_per_run": 0.10,
        "supports": ["image"],
    },
    "scratch_remover": {
        "model": "microsoft/bringing-old-photos-back-to-life",
        "version": "c75db81db6cbd809d93cc3b7e7a088a351a3349c9fa02b6d393e35e0d51ba799",
        "category": "photo_enhance",
        "description": "Remove scratches and damage",
        "cost_per_run": 0.10,
        "supports": ["image"],
    },
    "denoise": {
        "model": "sczhou/codeformer",
        "version": "cc4956dd26fa5a7185d5660cc9100fab1b8070a1d1654a8bb5eb6d443b020bb2",
        "category": "photo_enhance",
        "description": "Remove noise and grain",
        "cost_per_run": 0.10,
        "supports": ["image"],
    },
    "anime_yourself": {
        "model": "asiryan/meina-mix-v11",
        "version": "f0eba373c70464e12e48defa5520bef59f727018779afb9c5b6bddb80523a8f7",
        "category": "photo_enhance",
        "description": "Transform into anime style",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
        "default_prompt": "anime style portrait, beautiful detailed anime art, vibrant colors, masterpiece",
    },
    "portrait_mode": {
        "model": "lucataco/remove-bg",
        "version": "95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1",
        "category": "photo_enhance",
        "description": "Add professional bokeh effect",
        "cost_per_run": 0.10,
        "supports": ["image"],
        "client_side_processing": "blur_background",
    },
    "color_correct": {
        "model": "google-research/maxim",
        "version": None,  # Use latest version
        "category": "photo_enhance",
        "description": "Professional color grading",
        "cost_per_run": 0.10,
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
        "cost_per_run": 0.10,
        "supports": ["image", "mask"],
    },
    "ai_headshot": {
        "model": "tencentarc/photomaker",
        "version": "ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
        "category": "magic_edit",
        "description": "Professional headshots instantly",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
        "default_prompt": "professional headshot portrait, studio lighting, business photo, high quality, sharp focus",
    },
    "clothes_swap": {
        "model": "cuuupid/idm-vton",
        "version": None,  # Use latest version
        "category": "magic_edit",
        "description": "Try on different outfits",
        "cost_per_run": 0.10,
        "supports": ["image", "garment_image"],
    },
    "bg_remix": {
        "model": "black-forest-labs/flux-fill-schnell",
        "version": None,
        "category": "magic_edit",
        "description": "Change background creatively",
        "cost_per_run": 0.10,
        "supports": ["image", "mask", "prompt"],
    },
    "sticker_maker": {
        "model": "lucataco/remove-bg",
        "version": "95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1",
        "category": "magic_edit",
        "description": "Create custom stickers",
        "cost_per_run": 0.10,
        "supports": ["image"],
        "client_side_processing": "add_outline",
    },
    "outpainting": {
        "model": "black-forest-labs/flux-fill-schnell",
        "version": None,
        "category": "magic_edit",
        "description": "Expand image boundaries",
        "cost_per_run": 0.10,
        "supports": ["image", "mask", "prompt"],
    },
    "sky_replace": {
        "model": "black-forest-labs/flux-fill-schnell",
        "version": None,
        "category": "magic_edit",
        "description": "Change sky in photos",
        "cost_per_run": 0.10,
        "supports": ["image", "mask", "prompt"],
        "default_prompt": "beautiful dramatic sky, sunset clouds, golden hour, vibrant colors",
    },
    "interior_design": {
        "model": "adirik/interior-design",
        "version": "76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38",
        "category": "magic_edit",
        "description": "Redesign room interiors",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
        "default_prompt": "modern interior design, stylish furniture, professional photography",
    },
    "product_shoot": {
        "model": "zsxkib/ic-light",
        "version": "d41bcb10d8c159868f4cfbd7c6a2ca01484f7d39e4613419d5952c61562f1ba7",
        "category": "magic_edit",
        "description": "Professional product photos",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
        "default_prompt": "professional product photography, studio lighting, white background, commercial quality",
    },
    "text_effects": {
        "model": "jagilley/controlnet-canny",
        "version": "aff48af9c68d162388d230a2ab003f68d2638d88307bdaf1c2f1ac95079c9613",
        "category": "magic_edit",
        "description": "Add stylish text overlays",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
    },
    "tattoo_tryon": {
        "model": "black-forest-labs/flux-fill-schnell",
        "version": None,
        "category": "magic_edit",
        "description": "Preview tattoos on your body",
        "cost_per_run": 0.10,
        "supports": ["image", "mask", "prompt"],
    },
    # ─────────────────────────────────────────────────────────────────────
    # CREATIVE TOOLS
    # ─────────────────────────────────────────────────────────────────────
    "time_machine": {
        "model": "bytedance/flux-pulid",
        "version": "8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",
        "category": "magic_edit",
        "description": "See yourself at different ages - past or future",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
        "default_prompt": "professional portrait photo, high quality, detailed face",
        # TUNED for drastic transformations (aging/de-aging)
        "guidance_scale": 7.5,
        "id_weight": 0.8,
        "start_step": 2,
        "num_steps": 20,
    },
    "retro_style": {
        "model": "stability-ai/sdxl",
        "version": None,  # Use latest
        "category": "magic_edit",
        "description": "Retro & Polaroid aesthetic transformations",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
        "default_prompt": "vintage polaroid photo, retro aesthetic, film grain, warm faded colors",
        # SDXL img2img params - preserves full composition
        "prompt_strength": 0.75,  # 75% style, 25% original structure
        "num_inference_steps": 40,
        "guidance_scale": 7.5,
    },
    "style_transfer": {
        "model": "fofr/style-transfer",
        "version": "f1023890703bc0a5a3a2c21b5e498833be5f6ef6e70e9daf6b9b3a4fd8309cf0",
        "category": "magic_edit",
        "description": "Apply artistic styles to photos",
        "cost_per_run": 0.10,
        "supports": ["image", "style"],
        # Optimized params for face preservation
        "structure_denoising_strength": 0.5,  # Lower = better face preservation (0.5 ideal)
        "structure_depth_strength": 1.2,       # Max depth preservation for skeleton
    },
    "baby_prediction": {
        "model": "smoosh-sh/baby-mystic",
        "version": "ba5ab694a9df055fa469e55eeab162cc288039da0abd8b19d956980cc3b49f6d",
        "category": "magic_edit",
        "description": "Predict what your baby would look like",
        "cost_per_run": 0.10,
        "supports": ["image", "image2", "gender"],
        # Baby-Mystic params
        "steps": 25,
        "width": 512,
        "height": 728,
    },
    # ─────────────────────────────────────────────────────────────────────
    # TEXT-TO-IMAGE / IMAGE-TO-IMAGE GENERATORS
    # ─────────────────────────────────────────────────────────────────────
    "text_to_image": {
        "model": "black-forest-labs/flux-1.1-pro",
        "version": None,  # Use latest
        "category": "magic_edit",
        "description": "Generate images from text prompts with FLUX 1.1 Pro",
        "cost_per_run": 0.04,  # ~15 credits
        "supports": ["prompt"],
        # FLUX 1.1 Pro params
        "aspect_ratio": "1:1",
        "output_format": "webp",
        "output_quality": 90,
        "safety_tolerance": 2,
        "prompt_upsampling": True,
    },
    "image_to_image": {
        "model": "black-forest-labs/flux-dev",
        "version": None,  # Use latest
        "category": "magic_edit",
        "description": "Transform images with AI using FLUX Dev",
        "cost_per_run": 0.04,  # ~15 credits
        "supports": ["image", "prompt"],
        # FLUX Dev params
        "guidance": 3.5,
        "num_inference_steps": 28,
        "prompt_strength": 0.8,
        "output_format": "webp",
    },
    # img2img - alias for image_to_image (Flutter app uses this ID)
    "img2img": {
        "model": "black-forest-labs/flux-dev",
        "version": None,  # Use latest
        "category": "magic_edit",
        "description": "Transform images with AI using FLUX Dev",
        "cost_per_run": 0.04,  # ~15 credits
        "supports": ["image", "prompt"],
        # FLUX Dev params
        "guidance": 3.5,
        "num_inference_steps": 28,
        "prompt_strength": 0.8,
        "output_format": "webp",
    },
    "flux_pro": {
        "model": "black-forest-labs/flux-1.1-pro",
        "version": None,  # Use latest
        "category": "magic_edit",
        "description": "FLUX 1.1 Pro - High quality image generation with optional image guidance",
        "cost_per_run": 0.10,
        "supports": ["prompt", "image"],  # image_prompt for Redux mode
        # FLUX 1.1 Pro params
        "aspect_ratio": "1:1",  # 1:1, 16:9, 9:16, 4:3, 3:4, etc.
        "output_format": "webp",
        "output_quality": 90,
        "safety_tolerance": 2,  # 1=strict, 6=permissive
        "prompt_upsampling": False,
    },
    "sd35_large": {
        "model": "stability-ai/stable-diffusion-3.5-large",
        "version": None,  # Use latest
        "category": "magic_edit",
        "description": "Stable Diffusion 3.5 Large - Powerful image-to-image transformation",
        "cost_per_run": 0.10,
        "supports": ["prompt", "image", "negative_prompt"],
        # SD 3.5 Large params
        "cfg": 5,  # Guidance scale 1-10
        "prompt_strength": 0.85,  # 0-1, higher = more change from original
        "aspect_ratio": "1:1",
        "output_format": "webp",
    },
    # ─────────────────────────────────────────────────────────────────────
    # FLUX DEV LORA STYLE TRANSFERS (Retro/Vintage Effects)
    # ─────────────────────────────────────────────────────────────────────
    "vhs_style": {
        "model": "black-forest-labs/flux-dev-lora",
        "version": None,  # Use latest
        "category": "magic_edit",
        "description": "VHS retro effect - 90s analog glitch aesthetic",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
        # LoRA config
        "hf_lora": "huggingface.co/Muapi/vhx-the-flux-vhs-lora/vhx-the-flux-vhs-lora.safetensors",
        "lora_scale": 0.7,
        "prompt_strength": 0.35,
        "guidance_scale": 3.5,
        "default_prompt": "vhx style, VHS tape recording, analog glitch, 1990s footage, noise distortion",
        "output_format": "webp",
        "num_inference_steps": 28,
    },
    "polaroid_style": {
        "model": "black-forest-labs/flux-dev-lora",
        "version": None,  # Use latest
        "category": "magic_edit",
        "description": "Polaroid instant film - vintage flash photography aesthetic",
        "cost_per_run": 0.10,
        "supports": ["image", "prompt"],
        # LoRA config
        "hf_lora": "huggingface.co/prithivMLmods/Flux-Polaroid-Plus/polaroid-plus.safetensors",
        "lora_scale": 0.7,
        "prompt_strength": 0.35,
        "guidance_scale": 3.5,
        "default_prompt": "Polaroid Plus style, instant film photo, vintage polaroid frame, soft flash photography",
        "output_format": "webp",
        "num_inference_steps": 28,
    },
    # ─────────────────────────────────────────────────────────────────────
    # KLING V2.6 - High Quality Video Generation
    # ─────────────────────────────────────────────────────────────────────
    "kling_v26": {
        "model": "kwaivgi/kling-v2.6",
        "version": None,  # Use latest
        "category": "video",
        "description": "Kling v2.6 - High quality video generation with optional audio",
        "cost_per_run": 0.50,
        "supports": ["prompt", "image"],  # Both text-to-video and image-to-video
        "output_type": "video",
        # Kling v2.6 params - User requested 5s, 9:16, silent
        "duration": 5,  # 5 or 10 seconds
        "aspect_ratio": "9:16",  # 16:9, 9:16, 1:1
        "generate_audio": False,  # Silent videos
        "default_prompt": "cinematic video, smooth motion, high quality",
    },
    # ─────────────────────────────────────────────────────────────────────
    # GOOGLE VEO 3 FAST - Text-to-Video & Image-to-Video
    # ─────────────────────────────────────────────────────────────────────
    "veo_3_fast": {
        "model": "google/veo-3-fast",
        "version": None,  # Use latest
        "category": "video",
        "description": "Google Veo 3 Fast - High quality video generation from text or image",
        "cost_per_run": 0.50,
        "supports": ["prompt", "image"],  # Both text-to-video and image-to-video
        "output_type": "video",
        # Veo 3 params - User requested 720p and no audio
        "resolution": "720p",  # 720p or 1080p
        "duration": 4,  # 4, 6, or 8 seconds (4 = cheapest)
        "aspect_ratio": "9:16",  # 9:16 (vertical) or 16:9 (horizontal)
        "generate_audio": False,  # Silent videos as requested
        "default_prompt": "cinematic video, smooth motion, high quality",
    },
    # ─────────────────────────────────────────────────────────────────────
    # GOOGLE DEEPMIND - NANO BANANA PRO
    # ─────────────────────────────────────────────────────────────────────
    "nano_banana_pro": {
        "model": "google/nano-banana-pro",
        "version": None,  # Use latest
        "category": "magic_edit",
        "description": "Google DeepMind image generation with accurate text rendering and multi-image blending",
        "cost_per_run": 0.15,  # $0.15 for K/2K, $0.30 for 4K
        "supports": ["image", "prompt"],
        # Nano Banana Pro params - Resolution: "K" (1K), "2K", "4K"
        "resolution": "2K",  # Valid: K, 2K, 4K (API changed from 1K to K)
        "aspect_ratio": "match_input_image",  # 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
        "output_format": "jpg",  # jpg, png
        "safety_filter_level": "block_only_high",  # block_low_and_above, block_medium_and_above, block_only_high
    },
}

# Tool categories for frontend - All 33 Tools + 1 new
TOOL_CATEGORIES = {
    "VIDEO_AI": [
        "ai_hug", "image_to_video", "kling_video", "text_to_video", "talking_head",
        "video_expand", "style_transfer_video", "video_bg_remove", "face_swap_video",
        "script_to_video", "veo_3_fast", "kling_v26"
    ],
    "PHOTO_ENHANCE": [
        "face_clarify", "old_photo_restore", "colorize", "4k_upscale",
        "relight", "low_light_fix", "scratch_remover", "denoise",
        "anime_yourself", "portrait_mode", "color_correct"
    ],
    "MAGIC_EDIT": [
        "magic_eraser", "ai_headshot", "clothes_swap", "bg_remix",
        "sticker_maker", "outpainting", "sky_replace", "interior_design",
        "product_shoot", "text_effects", "tattoo_tryon", "style_transfer",
        "time_machine", "retro_style", "baby_prediction", "flux_pro", "sd35_large",
        "vhs_style", "polaroid_style", "nano_banana_pro", "text_to_image", "image_to_image"
    ]
}
