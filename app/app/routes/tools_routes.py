"""
AIVision API - Tools Routes
Endpoints for listing and getting tool information
"""
from fastapi import APIRouter, HTTPException
from typing import List

from ..models.schemas import ToolInfo, ToolsListResponse, ToolCategory
from ..config import AI_MODELS, TOOL_CATEGORIES
from ..services.replicate_service import replicate_service

router = APIRouter(prefix="/api/v1/tools", tags=["Tools"])


# Tool metadata for frontend - All 33 Tools matching Flutter app
TOOL_METADATA = {
    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO AI (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════
    "ai_hug": {
        "name": "AI Hug",
        "description": "Create heartwarming hug animations",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": 6,
        "example_prompt": None,
    },
    "image_to_video": {
        "name": "Image to Video",
        "description": "Animate your images with AI",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "image_text",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": 10,
        "example_prompt": "Gentle wind blowing, leaves moving, subtle camera pan",
    },
    "text_to_video": {
        "name": "Text to Video",
        "description": "Generate videos from text prompts",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "text_only",
        "supports_styles": True,
        "supports_aspect_ratio": True,
        "max_duration": 10,
        "example_prompt": "A serene lake at sunset with mountains in the background",
    },
    "talking_head": {
        "name": "Talking Head",
        "description": "Lip sync any portrait photo",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "image_audio",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": 60,
        "example_prompt": None,
    },
    "video_expand": {
        "name": "Video Expand",
        "description": "Extend video canvas seamlessly",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "video_only",
        "supports_styles": False,
        "supports_aspect_ratio": True,
        "max_duration": None,
        "example_prompt": None,
    },
    "style_transfer_video": {
        "name": "Style Transfer",
        "description": "Apply artistic styles to videos",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "video_only",
        "supports_styles": True,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "super_slowmo": {
        "name": "Super Slow-Mo",
        "description": "Create cinematic slow motion",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "video_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "video_upscale": {
        "name": "Video Upscale",
        "description": "Enhance video to 4K quality",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "video_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "video_bg_remove": {
        "name": "Video BG Remove",
        "description": "Remove video background",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "video_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "face_swap_video": {
        "name": "Face Swap Video",
        "description": "Swap faces in videos",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "image_video",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "script_to_video": {
        "name": "Script to Video",
        "description": "Turn scripts into videos",
        "category": ToolCategory.VIDEO_AI,
        "input_type": "text_only",
        "supports_styles": True,
        "supports_aspect_ratio": True,
        "max_duration": 30,
        "example_prompt": "Scene 1: A busy city street. Scene 2: Coffee cup steaming.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PHOTO ENHANCE (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════
    "face_clarify": {
        "name": "Face Clarify",
        "description": "Restore face details in photos",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "old_photo_restore": {
        "name": "Old Photo Restore",
        "description": "Bring old memories back to life",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "colorize": {
        "name": "Colorize",
        "description": "Add color to B&W photos",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "4k_upscale": {
        "name": "4K Upscale",
        "description": "Enhance images to 4K resolution",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "relight": {
        "name": "Relight",
        "description": "Change lighting in photos",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "low_light_fix": {
        "name": "Low Light Fix",
        "description": "Brighten dark photos naturally",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "scratch_remover": {
        "name": "Scratch Remover",
        "description": "Remove scratches and damage",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "denoise": {
        "name": "Denoise",
        "description": "Remove noise and grain",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "anime_yourself": {
        "name": "Anime Yourself",
        "description": "Transform into anime style",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": True,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "portrait_mode": {
        "name": "Portrait Mode",
        "description": "Add professional bokeh effect",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "color_correct": {
        "name": "Color Correct",
        "description": "Professional color grading",
        "category": ToolCategory.PHOTO_ENHANCE,
        "input_type": "image_only",
        "supports_styles": True,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MAGIC EDIT (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════
    "magic_eraser": {
        "name": "Magic Eraser",
        "description": "Remove unwanted objects",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "image_mask",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "ai_headshot": {
        "name": "AI Headshot",
        "description": "Professional headshots instantly",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "image_text",
        "supports_styles": True,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": "Professional business headshot, studio lighting",
    },
    "clothes_swap": {
        "name": "Clothes Swap",
        "description": "Try on different outfits",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "dual_image",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "bg_remix": {
        "name": "BG Remix",
        "description": "Change background creatively",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "image_text",
        "supports_styles": True,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": "Beautiful sunset beach with palm trees",
    },
    "sticker_maker": {
        "name": "Sticker Maker",
        "description": "Create custom stickers",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "image_only",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "outpainting": {
        "name": "Outpainting",
        "description": "Expand image boundaries",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "image_text",
        "supports_styles": False,
        "supports_aspect_ratio": True,
        "max_duration": None,
        "example_prompt": "Continue the natural scenery seamlessly",
    },
    "sky_replace": {
        "name": "Sky Replace",
        "description": "Change sky in photos",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "image_text",
        "supports_styles": True,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": "Beautiful dramatic sunset sky with clouds",
    },
    "interior_design": {
        "name": "Interior Design",
        "description": "Redesign room interiors",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "image_text",
        "supports_styles": True,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": "Modern minimalist interior with natural lighting",
    },
    "product_shoot": {
        "name": "Product Shoot",
        "description": "Professional product photos",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "image_text",
        "supports_styles": True,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": "Professional product photo, studio lighting, white background",
    },
    "text_effects": {
        "name": "Text Effects",
        "description": "Add stylish text overlays",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "image_text",
        "supports_styles": True,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
    "tattoo_tryon": {
        "name": "Tattoo Try-on",
        "description": "Preview tattoos on your body",
        "category": ToolCategory.MAGIC_EDIT,
        "input_type": "dual_image",
        "supports_styles": False,
        "supports_aspect_ratio": False,
        "max_duration": None,
        "example_prompt": None,
    },
}


@router.get("", response_model=ToolsListResponse)
async def list_tools():
    """Get list of all available AI tools."""
    tools = []

    for tool_id, metadata in TOOL_METADATA.items():
        credit_cost = replicate_service.get_credit_cost(tool_id)

        tool = ToolInfo(
            tool_id=tool_id,
            name=metadata["name"],
            description=metadata["description"],
            category=metadata["category"],
            credit_cost=credit_cost,
            input_type=metadata["input_type"],
            supports_styles=metadata["supports_styles"],
            supports_aspect_ratio=metadata["supports_aspect_ratio"],
            max_duration=metadata.get("max_duration"),
            example_prompt=metadata.get("example_prompt"),
        )
        tools.append(tool)

    return ToolsListResponse(
        success=True,
        tools=tools,
        total_count=len(tools),
    )


@router.get("/{tool_id}", response_model=ToolInfo)
async def get_tool(tool_id: str):
    """Get information about a specific tool."""
    if tool_id not in TOOL_METADATA:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")

    metadata = TOOL_METADATA[tool_id]
    credit_cost = replicate_service.get_credit_cost(tool_id)

    return ToolInfo(
        tool_id=tool_id,
        name=metadata["name"],
        description=metadata["description"],
        category=metadata["category"],
        credit_cost=credit_cost,
        input_type=metadata["input_type"],
        supports_styles=metadata["supports_styles"],
        supports_aspect_ratio=metadata["supports_aspect_ratio"],
        max_duration=metadata.get("max_duration"),
        example_prompt=metadata.get("example_prompt"),
    )


@router.get("/category/{category}")
async def list_tools_by_category(category: str):
    """Get tools filtered by category."""
    try:
        cat = ToolCategory(category.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {category}. Valid: VIDEO_AI, PHOTO_ENHANCE, MAGIC_EDIT"
        )

    tools = []
    for tool_id, metadata in TOOL_METADATA.items():
        if metadata["category"] == cat:
            credit_cost = replicate_service.get_credit_cost(tool_id)
            tool = ToolInfo(
                tool_id=tool_id,
                name=metadata["name"],
                description=metadata["description"],
                category=metadata["category"],
                credit_cost=credit_cost,
                input_type=metadata["input_type"],
                supports_styles=metadata["supports_styles"],
                supports_aspect_ratio=metadata["supports_aspect_ratio"],
                max_duration=metadata.get("max_duration"),
                example_prompt=metadata.get("example_prompt"),
            )
            tools.append(tool)

    return {
        "success": True,
        "category": category,
        "tools": tools,
        "total_count": len(tools),
    }
