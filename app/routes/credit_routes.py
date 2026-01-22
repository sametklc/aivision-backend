"""
AIVision API - Credit Routes
Secure credit management with Firebase Auth verification
"""
from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import firebase_admin
from firebase_admin import auth, firestore, credentials
from datetime import datetime
import os
import json

router = APIRouter(prefix="/api/v1/credits", tags=["Credits"])


def _ensure_firebase_initialized():
    """Ensure Firebase Admin SDK is initialized."""
    try:
        firebase_admin.get_app()
    except ValueError:
        # Not initialized yet, initialize it
        firebase_creds_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

        if firebase_creds_json:
            try:
                creds_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(creds_dict)
                bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET", "aivision-47fb4.firebasestorage.app")
                firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
                logger.info("Firebase Admin SDK initialized for credit routes")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase: {e}")
                raise
        else:
            # Try file-based credentials
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


# ==================== MODELS ====================

class CreditBalanceResponse(BaseModel):
    success: bool
    credits: int
    user_id: str


class DeductCreditsRequest(BaseModel):
    amount: int = Field(..., gt=0, description="Amount to deduct")
    reason: str = Field(..., description="Reason for deduction (tool_id)")


class DeductCreditsResponse(BaseModel):
    success: bool
    credits_before: int
    credits_after: int
    amount_deducted: int


class AddCreditsRequest(BaseModel):
    amount: int = Field(..., gt=0, description="Amount to add")
    reason: str = Field(default="purchase", description="Reason for addition")


class AddCreditsResponse(BaseModel):
    success: bool
    credits_before: int
    credits_after: int
    amount_added: int


# ==================== AUTH HELPER ====================

async def verify_firebase_token(authorization: str = Header(...)) -> str:
    """
    Firebase ID Token'ı doğrula ve user_id döndür
    Header format: "Bearer <token>"
    """
    _ensure_firebase_initialized()  # Firebase must be initialized before auth

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.replace("Bearer ", "")

    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        logger.info(f"🔐 Token verified for user: {user_id}")
        return user_id
    except Exception as e:
        logger.error(f"❌ Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ==================== ENDPOINTS ====================

@router.get("/balance", response_model=CreditBalanceResponse)
async def get_balance(user_id: str = Depends(verify_firebase_token)):
    """
    Kullanıcının kredi bakiyesini getir
    """
    try:
        db = get_firestore_client()
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            # Yeni kullanıcı - 15 kredi ile oluştur
            initial_credits = 15
            user_ref.set({
                'id': user_id,
                'credits': initial_credits,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
            })
            logger.info(f"✅ Created new user {user_id} with {initial_credits} credits")
            return CreditBalanceResponse(success=True, credits=initial_credits, user_id=user_id)

        credits = user_doc.to_dict().get('credits', 0)
        return CreditBalanceResponse(success=True, credits=credits, user_id=user_id)

    except Exception as e:
        logger.error(f"❌ Error getting balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deduct", response_model=DeductCreditsResponse)
async def deduct_credits(
    request: DeductCreditsRequest,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Kredi düş (generation başlamadan önce çağrılır)
    """
    try:
        db = get_firestore_client()
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")

        current_credits = user_doc.to_dict().get('credits', 0)

        if current_credits < request.amount:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient credits. Have: {current_credits}, Need: {request.amount}"
            )

        new_credits = current_credits - request.amount

        # Update credits
        user_ref.update({
            'credits': new_credits,
            'updated_at': firestore.SERVER_TIMESTAMP,
        })

        # Log to credit_history
        user_ref.collection('credit_history').add({
            'amount': -request.amount,
            'reason': request.reason,
            'balance_after': new_credits,
            'created_at': firestore.SERVER_TIMESTAMP,
        })

        logger.info(f"💰 Deducted {request.amount} credits from {user_id}. {current_credits} -> {new_credits}")

        return DeductCreditsResponse(
            success=True,
            credits_before=current_credits,
            credits_after=new_credits,
            amount_deducted=request.amount
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deducting credits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add", response_model=AddCreditsResponse)
async def add_credits(
    request: AddCreditsRequest,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Kredi ekle (satın alma sonrası veya refund için)
    """
    try:
        db = get_firestore_client()
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            # Yeni kullanıcı oluştur
            current_credits = 0
            new_credits = request.amount
            user_ref.set({
                'id': user_id,
                'credits': new_credits,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
            })
        else:
            current_credits = user_doc.to_dict().get('credits', 0)
            new_credits = current_credits + request.amount
            user_ref.update({
                'credits': new_credits,
                'updated_at': firestore.SERVER_TIMESTAMP,
            })

        # Log to credit_history
        user_ref.collection('credit_history').add({
            'amount': request.amount,
            'reason': request.reason,
            'balance_after': new_credits,
            'created_at': firestore.SERVER_TIMESTAMP,
        })

        logger.info(f"💰 Added {request.amount} credits to {user_id}. {current_credits} -> {new_credits}")

        return AddCreditsResponse(
            success=True,
            credits_before=current_credits,
            credits_after=new_credits,
            amount_added=request.amount
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error adding credits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refund", response_model=AddCreditsResponse)
async def refund_credits(
    request: AddCreditsRequest,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Kredi iade et (generation başarısız olduğunda)
    """
    request.reason = f"refund_{request.reason}"
    return await add_credits(request, user_id)
