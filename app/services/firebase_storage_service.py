"""
AIVision - Firebase Storage Service
Handles video/image uploads to Firebase Storage (replacing Cloudinary)
"""
import firebase_admin
from firebase_admin import credentials, storage
from typing import Optional, Tuple
from loguru import logger
import uuid
import tempfile
import os
import base64
import json

from ..config import get_settings


class FirebaseStorageService:
    """Service for uploading files to Firebase Storage."""

    def __init__(self):
        self._initialized = False
        self._bucket = None
        self._init_firebase()

    def _init_firebase(self):
        """Initialize Firebase Admin SDK."""
        try:
            settings = get_settings()

            # Check if already initialized
            try:
                firebase_admin.get_app()
                self._initialized = True
                self._bucket = storage.bucket()
                logger.info("Firebase already initialized, reusing existing app")
                return
            except ValueError:
                pass  # Not initialized yet

            # Try to get credentials from environment variable
            firebase_creds_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

            if firebase_creds_json:
                # Parse JSON from environment variable
                try:
                    creds_dict = json.loads(firebase_creds_json)
                    cred = credentials.Certificate(creds_dict)

                    # Get bucket name from env or use default
                    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET", "aivision-47fb4.firebasestorage.app")

                    firebase_admin.initialize_app(cred, {
                        'storageBucket': bucket_name
                    })
                    self._bucket = storage.bucket()
                    self._initialized = True
                    logger.info(f"Firebase Admin SDK initialized with bucket: {bucket_name}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse FIREBASE_SERVICE_ACCOUNT JSON: {e}")
                except Exception as e:
                    logger.error(f"Failed to initialize Firebase with env credentials: {e}")
            else:
                # Try to find service account file
                service_account_path = os.environ.get(
                    "FIREBASE_SERVICE_ACCOUNT_PATH",
                    "firebase-service-account.json"
                )

                if os.path.exists(service_account_path):
                    cred = credentials.Certificate(service_account_path)
                    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET", "aivision-47fb4.firebasestorage.app")

                    firebase_admin.initialize_app(cred, {
                        'storageBucket': bucket_name
                    })
                    self._bucket = storage.bucket()
                    self._initialized = True
                    logger.info(f"Firebase Admin SDK initialized from file: {service_account_path}")
                else:
                    logger.warning("Firebase credentials not found. Set FIREBASE_SERVICE_ACCOUNT or FIREBASE_SERVICE_ACCOUNT_PATH")

        except Exception as e:
            logger.error(f"Firebase initialization error: {e}")
            self._initialized = False

    @property
    def is_configured(self) -> bool:
        return self._initialized and self._bucket is not None

    async def upload_video(
        self,
        video_data: bytes,
        folder: str = "videos",
        filename: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload video to Firebase Storage.

        Args:
            video_data: Raw video bytes
            folder: Storage folder path
            filename: Custom filename (optional)

        Returns:
            Tuple of (success, url, error)
        """
        if not self.is_configured:
            logger.warning("Firebase Storage not configured")
            return False, None, "Firebase Storage not configured"

        temp_path = None
        try:
            # Generate unique filename if not provided
            if not filename:
                filename = f"{uuid.uuid4().hex[:16]}.mp4"

            blob_path = f"{folder}/{filename}"
            logger.info(f"Uploading video to Firebase Storage: {blob_path} ({len(video_data)} bytes)")

            # Save to temp file first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(video_data)
                temp_path = temp_file.name

            # Upload to Firebase Storage
            blob = self._bucket.blob(blob_path)
            blob.upload_from_filename(temp_path, content_type='video/mp4')

            # Make the file publicly accessible
            blob.make_public()

            url = blob.public_url
            logger.info(f"Video uploaded to Firebase: {url}")
            return True, url, None

        except Exception as e:
            logger.error(f"Firebase video upload error: {str(e)}")
            return False, None, str(e)
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    async def upload_image(
        self,
        image_data: bytes,
        folder: str = "images",
        filename: Optional[str] = None,
        content_type: str = "image/jpeg"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload image to Firebase Storage.

        Args:
            image_data: Raw image bytes
            folder: Storage folder path
            filename: Custom filename (optional)
            content_type: Image content type

        Returns:
            Tuple of (success, url, error)
        """
        if not self.is_configured:
            logger.warning("Firebase Storage not configured")
            return False, None, "Firebase Storage not configured"

        temp_path = None
        try:
            # Determine extension from content type
            ext = "jpg"
            if "png" in content_type:
                ext = "png"
            elif "webp" in content_type:
                ext = "webp"

            # Generate unique filename if not provided
            if not filename:
                filename = f"{uuid.uuid4().hex[:16]}.{ext}"

            blob_path = f"{folder}/{filename}"
            logger.info(f"Uploading image to Firebase Storage: {blob_path} ({len(image_data)} bytes)")

            # Save to temp file first
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name

            # Upload to Firebase Storage
            blob = self._bucket.blob(blob_path)
            blob.upload_from_filename(temp_path, content_type=content_type)

            # Make the file publicly accessible
            blob.make_public()

            url = blob.public_url
            logger.info(f"Image uploaded to Firebase: {url}")
            return True, url, None

        except Exception as e:
            logger.error(f"Firebase image upload error: {str(e)}")
            return False, None, str(e)
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass


# Global service instance
firebase_storage_service = FirebaseStorageService()
