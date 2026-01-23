"""
AIVision - Upload Routes
Handles image upload to Cloudinary
SECURED: All uploads require Firebase Auth
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Header
from typing import Optional
from loguru import logger
import firebase_admin
from firebase_admin import auth, credentials
import os
import json

from ..services.cloudinary_service import cloudinary_service

router = APIRouter(prefix="/api/v1/upload", tags=["Upload"])


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
                logger.info("Firebase Admin SDK initialized for upload routes")
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
            else:
                raise ValueError("Firebase credentials not found")


async def verify_firebase_token(authorization: str = Header(...)) -> str:
    """Verify Firebase ID Token and return user_id."""
    _ensure_firebase_initialized()

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.replace("Bearer ", "")

    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        logger.info(f"🔐 Upload - Token verified for user: {user_id}")
        return user_id
    except Exception as e:
        logger.error(f"❌ Upload - Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("")
async def upload_image(
    file: UploadFile = File(...),
    folder: Optional[str] = Form(default="aivision"),
    user_id: str = Depends(verify_firebase_token)
):
    """
    Upload image to Cloudinary CDN.
    SECURED: Requires Firebase Auth.
    Returns the public URL of the uploaded image.
    """
    logger.info(f"📤 Image upload by user: {user_id}")
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
    folder: Optional[str] = Form(default="aivision"),
    user_id: str = Depends(verify_firebase_token)
):
    """
    Upload base64 encoded image to Cloudinary.
    SECURED: Requires Firebase Auth.

    Args:
        image: Base64 encoded image string
        folder: Cloudinary folder name
    """
    logger.info(f"📤 Base64 upload by user: {user_id}")
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


@router.post("/video")
async def upload_video(
    file: UploadFile = File(...),
    folder: Optional[str] = Form(default="aivision/videos"),
    user_id: str = Depends(verify_firebase_token)
):
    """
    Upload video to Cloudinary CDN.
    SECURED: Requires Firebase Auth.

    Returns the public URL of the uploaded video.
    Accepts MP4, MOV, AVI, MKV, and WebM formats.
    Max size: 100MB
    """
    logger.info(f"📤 Video upload by user: {user_id}")
    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska", "video/webm"]
    if not file.content_type or file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File must be a video (mp4, mov, avi, mkv, webm). Got: {file.content_type}"
        )

    # Check file size (max 100MB for videos)
    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(
            status_code=400,
            detail="Video file size must be less than 100MB"
        )

    # Check if Cloudinary is configured
    if not cloudinary_service.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Video upload service not configured"
        )

    # Upload to Cloudinary
    success, url, error = await cloudinary_service.upload_video(
        video_data=contents,
        folder=folder
    )

    if not success:
        logger.error(f"Video upload failed: {error}")
        raise HTTPException(
            status_code=500,
            detail=f"Video upload failed: {error}"
        )

    return {
        "success": True,
        "url": url,
        "filename": file.filename,
        "size": len(contents)
    }
