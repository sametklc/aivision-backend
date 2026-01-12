"""
AIVision - Cloudinary Service
Handles image uploads to Cloudinary CDN
"""
import cloudinary
import cloudinary.uploader
from typing import Optional, Tuple
from loguru import logger
import base64
import uuid

from ..config import get_settings


class CloudinaryService:
    """Service for uploading images to Cloudinary."""

    def __init__(self):
        settings = get_settings()
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True
        )
        self._configured = bool(
            settings.CLOUDINARY_CLOUD_NAME and
            settings.CLOUDINARY_API_KEY and
            settings.CLOUDINARY_API_SECRET
        )

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def upload_image(
        self,
        image_data: bytes,
        folder: str = "aivision",
        public_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload image to Cloudinary.

        Args:
            image_data: Raw image bytes
            folder: Cloudinary folder name
            public_id: Custom public ID (optional)

        Returns:
            Tuple of (success, url, error)
        """
        if not self._configured:
            logger.warning("Cloudinary not configured")
            return False, None, "Cloudinary not configured"

        try:
            # Generate unique ID if not provided
            if not public_id:
                public_id = f"{folder}/{uuid.uuid4().hex[:12]}"

            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                image_data,
                public_id=public_id,
                folder=folder,
                resource_type="image",
                overwrite=True,
                transformation=[
                    {"quality": "auto:good"},
                    {"fetch_format": "auto"}
                ]
            )

            url = result.get("secure_url")
            logger.info(f"Image uploaded to Cloudinary: {url}")
            return True, url, None

        except Exception as e:
            logger.error(f"Cloudinary upload error: {str(e)}")
            return False, None, str(e)

    async def upload_base64(
        self,
        base64_data: str,
        folder: str = "aivision"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload base64 encoded image to Cloudinary.

        Args:
            base64_data: Base64 encoded image string
            folder: Cloudinary folder name

        Returns:
            Tuple of (success, url, error)
        """
        if not self._configured:
            logger.warning("Cloudinary not configured")
            return False, None, "Cloudinary not configured"

        try:
            # Handle data URI format
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]

            # Upload directly with base64
            result = cloudinary.uploader.upload(
                f"data:image/jpeg;base64,{base64_data}",
                folder=folder,
                resource_type="image",
                transformation=[
                    {"quality": "auto:good"},
                    {"fetch_format": "auto"}
                ]
            )

            url = result.get("secure_url")
            logger.info(f"Base64 image uploaded to Cloudinary: {url}")
            return True, url, None

        except Exception as e:
            logger.error(f"Cloudinary base64 upload error: {str(e)}")
            return False, None, str(e)

    async def upload_video(
        self,
        video_data: bytes,
        folder: str = "aivision/videos",
        public_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload video to Cloudinary.

        Args:
            video_data: Raw video bytes
            folder: Cloudinary folder name
            public_id: Custom public ID (optional)

        Returns:
            Tuple of (success, url, error)
        """
        if not self._configured:
            logger.warning("Cloudinary not configured")
            return False, None, "Cloudinary not configured"

        try:
            # Generate unique ID if not provided
            if not public_id:
                public_id = f"{folder}/{uuid.uuid4().hex[:12]}"

            logger.info(f"Uploading video to Cloudinary ({len(video_data)} bytes)...")

            # Upload to Cloudinary as video resource
            result = cloudinary.uploader.upload(
                video_data,
                public_id=public_id,
                folder=folder,
                resource_type="video",  # Important: video resource type
                overwrite=True,
            )

            url = result.get("secure_url")
            logger.info(f"Video uploaded to Cloudinary: {url}")
            return True, url, None

        except Exception as e:
            logger.error(f"Cloudinary video upload error: {str(e)}")
            return False, None, str(e)


# Global service instance
cloudinary_service = CloudinaryService()
