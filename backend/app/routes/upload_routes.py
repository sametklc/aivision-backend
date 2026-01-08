"""
AIVision - Upload Routes
Handles image upload to Cloudinary
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from loguru import logger

from ..services.cloudinary_service import cloudinary_service

router = APIRouter(prefix="/api/v1/upload", tags=["Upload"])


@router.post("")
async def upload_image(
    file: UploadFile = File(...),
    folder: Optional[str] = Form(default="aivision")
):
    """
    Upload image to Cloudinary CDN.

    Returns the public URL of the uploaded image.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    # Check file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400,
            detail="File size must be less than 10MB"
        )

    # Check if Cloudinary is configured
    if not cloudinary_service.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Image upload service not configured"
        )

    # Upload to Cloudinary
    success, url, error = await cloudinary_service.upload_image(
        image_data=contents,
        folder=folder
    )

    if not success:
        logger.error(f"Upload failed: {error}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {error}"
        )

    return {
        "success": True,
        "url": url,
        "filename": file.filename,
        "size": len(contents)
    }


@router.post("/base64")
async def upload_base64(
    image: str = Form(...),
    folder: Optional[str] = Form(default="aivision")
):
    """
    Upload base64 encoded image to Cloudinary.

    Args:
        image: Base64 encoded image string
        folder: Cloudinary folder name
    """
    if not cloudinary_service.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Image upload service not configured"
        )

    success, url, error = await cloudinary_service.upload_base64(
        base64_data=image,
        folder=folder
    )

    if not success:
        logger.error(f"Base64 upload failed: {error}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {error}"
        )

    return {
        "success": True,
        "url": url
    }
