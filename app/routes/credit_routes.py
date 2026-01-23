"""
AIVision API - Credit Routes
Secure credit management with Firebase Auth + RevenueCat validation
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
import httpx

router = APIRouter(prefix="/api/v1/credits", tags=["Credits"])

# RevenueCat API configuration
REVENUECAT_API_KEY = os.environ.get("REVENUECAT_API_KEY", "")
REVENUECAT_BASE_URL = "https://api.revenuecat.com/v1"

# Internal API key for backend-only endpoints
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")


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
                logger.info("Firebase Admin SDK initialized for credit routes")
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
    transaction_id: str = Field(..., description="RevenueCat transaction ID for validation")
    product_id: str = Field(..., description="Product ID that was purchased")


class AddCreditsResponse(BaseModel):
    success: bool
    credits_before: int
    credits_after: int
    amount_added: int


class InternalRefundRequest(BaseModel):
    user_id: str = Field(..., description="User ID to refund")
    amount: int = Field(..., gt=0, description="Amount to refund")
    reason: str = Field(..., description="Reason for refund")
    job_id: Optional[str] = Field(None, description="Failed job ID")


# ==================== AUTH HELPERS ====================

async def verify_firebase_token(authorization: str = Header(...)) -> str:
    """
    Firebase ID Token'ı doğrula ve user_id döndür
    Header format: "Bearer <token>"
    """
    _ensure_firebase_initialized()

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


async def verify_internal_api_key(x_internal_key: str = Header(..., alias="X-Internal-Key")) -> bool:
    """
    Verify internal API key for backend-only endpoints
    """
    if not INTERNAL_API_KEY:
        logger.error("❌ INTERNAL_API_KEY not configured!")
        raise HTTPException(status_code=500, detail="Internal API key not configured")

    if x_internal_key != INTERNAL_API_KEY:
        logger.warning("⚠️ Invalid internal API key attempt")
        raise HTTPException(status_code=403, detail="Invalid internal API key")

    return True


# ==================== REVENUECAT VALIDATION ====================

async def validate_revenuecat_purchase(user_id: str, transaction_id: str, product_id: str) -> bool:
    """
    Validate purchase with RevenueCat API
    Returns True if purchase is valid and not already processed
    """
    if not REVENUECAT_API_KEY:
        logger.error("❌ REVENUECAT_API_KEY not configured!")
        raise HTTPException(status_code=500, detail="RevenueCat API key not configured")

    # Use Firebase user ID as RevenueCat user ID (they should match after logIn)
    # Also try extracted ID from transaction as fallback for older purchases
    rc_user_id = user_id
    extracted_rc_user_id = None

    if "_" in transaction_id:
        parts = transaction_id.split("_")
        if len(parts) >= 2:
            extracted_rc_user_id = parts[0]
            logger.info(f"🔍 Transaction contains user ID: {extracted_rc_user_id}, Firebase user: {user_id}")

    # List of user IDs to try (Firebase first, then extracted if different)
    user_ids_to_try = [user_id]
    if extracted_rc_user_id and extracted_rc_user_id != user_id:
        user_ids_to_try.append(extracted_rc_user_id)

    logger.info(f"🔍 Will try RevenueCat user IDs: {user_ids_to_try}")

    try:
        async with httpx.AsyncClient() as client:
            # Try each user ID until we find a valid purchase
            for rc_user_id in user_ids_to_try:
                logger.info(f"🔍 Trying RevenueCat user ID: {rc_user_id}")

                response = await client.get(
                    f"{REVENUECAT_BASE_URL}/subscribers/{rc_user_id}",
                    headers={
                        "Authorization": f"Bearer {REVENUECAT_API_KEY}",
                        "Content-Type": "application/json",
                    }
                )

                if response.status_code == 404:
                    logger.warning(f"⚠️ RevenueCat subscriber not found: {rc_user_id}")
                    continue  # Try next user ID

                # Accept both 200 (existing subscriber) and 201 (new subscriber created)
                if response.status_code not in [200, 201]:
                    logger.error(f"❌ RevenueCat API error: {response.status_code} - {response.text}")
                    continue  # Try next user ID

                data = response.json()
                subscriber = data.get("subscriber", {})

                # Check non_subscriptions for consumables (credit packages)
                non_subscriptions = subscriber.get("non_subscriptions", {})

                # Look for the product and transaction - also check for ANY purchase of this product
                product_purchases = non_subscriptions.get(product_id, [])

                # For consumables, if ANY purchase exists for this product, it's valid
                # (We'll use transaction_id for deduplication separately)
                if product_purchases:
                    logger.info(f"✅ Valid consumable purchase found for product {product_id}, user {rc_user_id}")
                    return True

                # Also check exact transaction match
                for purchase in product_purchases:
                    if purchase.get("id") == transaction_id or purchase.get("store_transaction_id") == transaction_id:
                        logger.info(f"✅ Valid purchase found: {transaction_id} for product {product_id}")
                        return True

                # Check subscriptions - for subscriptions, validate ANY active subscription
                subscriptions = subscriber.get("subscriptions", {})

                # Determine if this is a subscription product request
                is_subscription_product = any(
                    keyword in product_id.lower()
                    for keyword in ['weekly', 'yearly', 'monthly', 'pro', 'premium', 'subscription']
                )

                if is_subscription_product and subscriptions:
                    # For subscription products, ANY active subscription is valid
                    # This handles cases where product IDs don't match exactly
                    for sub_product_id, sub_info in subscriptions.items():
                        expires_date = sub_info.get("expires_date")
                        store_txn = sub_info.get("store_transaction_id")
                        original_date = sub_info.get("original_purchase_date")

                        logger.info(f"🔍 Found subscription {sub_product_id}: expires={expires_date}")

                        # Subscription is valid if it has any purchase record
                        if store_txn or original_date:
                            logger.info(f"✅ Valid subscription found: {sub_product_id} (requested: {product_id})")
                            return True

                # Also check specific product variants for exact match
                product_variants = [product_id]
                if ":" in product_id:
                    product_variants.append(product_id.split(":")[0])
                else:
                    product_variants.extend([
                        f"{product_id}:weekly",
                        f"{product_id}:yearly",
                        f"{product_id}:monthly",
                    ])

                for variant in product_variants:
                    if variant in subscriptions:
                        sub_info = subscriptions[variant]
                        if sub_info.get("store_transaction_id") or sub_info.get("original_purchase_date"):
                            logger.info(f"✅ Valid subscription found for product {variant}")
                            return True

                # Check entitlements as fallback - ANY active entitlement for subscription products
                entitlements = subscriber.get("entitlements", {})
                if is_subscription_product and entitlements:
                    for ent_name, ent_info in entitlements.items():
                        if ent_info.get("is_active") or ent_info.get("product_identifier"):
                            logger.info(f"✅ Valid entitlement '{ent_name}' found (requested: {product_id})")
                            return True

                # Check specific entitlement product match
                for ent_name, ent_info in entitlements.items():
                    if ent_info.get("product_identifier") in product_variants:
                        logger.info(f"✅ Valid entitlement '{ent_name}' found for product {product_id}")
                        return True

                logger.info(f"ℹ️ No matching purchase for user {rc_user_id}")
                logger.info(f"ℹ️ Available subscriptions: {list(subscriptions.keys())}")
                logger.info(f"ℹ️ Available non_subscriptions: {list(non_subscriptions.keys())}")

            # All user IDs tried, no valid purchase found
            logger.warning(f"⚠️ Purchase not found for any user ID: {transaction_id} for product {product_id}")
            raise HTTPException(status_code=400, detail="Purchase not found or already processed")

    except httpx.RequestError as e:
        logger.error(f"❌ RevenueCat request error: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to RevenueCat")


async def check_transaction_already_processed(user_id: str, transaction_id: str) -> bool:
    """
    Check if this transaction was already processed (prevent double-crediting)
    """
    db = get_firestore_client()

    # Check credit_history for this transaction
    history_ref = db.collection('users').document(user_id).collection('credit_history')
    query = history_ref.where('transaction_id', '==', transaction_id).limit(1)
    docs = query.get()

    return len(list(docs)) > 0


# ==================== CREDIT AMOUNTS FOR PRODUCTS ====================

PRODUCT_CREDITS = {
    # Credit packages (consumables)
    "com.aivision.credits.500": 500,
    "com.aivision.credits.1000": 1000,
    "com.aivision.credits.2000": 2000,
    # Subscriptions
    "com.aivision.weekly": 500,        # 1 week premium + 500 credits
    "com.aivision.weekly.pro": 1000,   # 1 week pro + 1000 credits
    "com.aivision.yearly": 4000,       # 1 year subscription + 4000 credits
}


# ==================== DEVICE TRACKING ====================

async def check_device_abuse(db, device_id: str, user_id: str) -> bool:
    """
    Check if device has already received free credits with a different account.
    Returns True if abuse detected (device already used free credits).
    """
    if not device_id or device_id == "unknown":
        return False  # Can't check without device ID

    # Check device_registry collection
    device_ref = db.collection('device_registry').document(device_id)
    device_doc = device_ref.get()

    if device_doc.exists:
        registered_user = device_doc.to_dict().get('user_id')
        if registered_user and registered_user != user_id:
            logger.warning(f"🚫 DEVICE ABUSE DETECTED! Device {device_id} already registered to {registered_user}, new user {user_id} trying to get free credits")
            return True

    return False


async def register_device(db, device_id: str, user_id: str):
    """Register device ID with user to prevent abuse."""
    if not device_id or device_id == "unknown":
        return

    device_ref = db.collection('device_registry').document(device_id)
    device_ref.set({
        'user_id': user_id,
        'registered_at': firestore.SERVER_TIMESTAMP,
    }, merge=True)
    logger.info(f"📱 Registered device {device_id} to user {user_id}")


# ==================== ENDPOINTS ====================

@router.get("/balance", response_model=CreditBalanceResponse)
async def get_balance(
    user_id: str = Depends(verify_firebase_token),
    device_id: str = Header(None, alias="X-Device-ID")
):
    """
    Kullanıcının kredi bakiyesini getir.
    Device ID ile anonim hesap abuse kontrolü yapılır.
    """
    try:
        db = get_firestore_client()
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            # Check for device abuse BEFORE creating new user
            if device_id:
                is_abuse = await check_device_abuse(db, device_id, user_id)
                if is_abuse:
                    # Device already used free credits - create user with 0 credits
                    logger.warning(f"🚫 Blocking free credits for abusive device: {device_id}")
                    user_ref.set({
                        'id': user_id,
                        'credits': 0,  # NO FREE CREDITS FOR ABUSERS
                        'device_id': device_id,
                        'abuse_detected': True,
                        'created_at': firestore.SERVER_TIMESTAMP,
                        'updated_at': firestore.SERVER_TIMESTAMP,
                    })
                    return CreditBalanceResponse(success=True, credits=0, user_id=user_id)

            # New legitimate user - 15 credits
            initial_credits = 15
            user_ref.set({
                'id': user_id,
                'credits': initial_credits,
                'device_id': device_id,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
            })

            # Register device to prevent abuse
            if device_id:
                await register_device(db, device_id, user_id)

            logger.info(f"✅ Created new user {user_id} with {initial_credits} credits (device: {device_id})")
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
            'type': 'deduct',
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
    Kredi ekle - SADECE DOĞRULANMIŞ SATIN ALMALAR İÇİN
    RevenueCat ile satın alma doğrulaması yapılır
    """
    try:
        # 1. Check if transaction already processed (prevent double-crediting)
        if await check_transaction_already_processed(user_id, request.transaction_id):
            logger.warning(f"⚠️ Transaction already processed: {request.transaction_id}")
            raise HTTPException(status_code=400, detail="Transaction already processed")

        # 2. Validate with RevenueCat
        await validate_revenuecat_purchase(user_id, request.transaction_id, request.product_id)

        # 3. Get credit amount for this product
        credit_amount = PRODUCT_CREDITS.get(request.product_id)
        if credit_amount is None:
            logger.error(f"❌ Unknown product ID: {request.product_id}")
            raise HTTPException(status_code=400, detail="Unknown product ID")

        # Override with request amount if it matches (for flexibility)
        if request.amount != credit_amount:
            logger.warning(f"⚠️ Request amount {request.amount} doesn't match product credits {credit_amount}")
            # Use the predefined amount for security
            credit_amount = PRODUCT_CREDITS[request.product_id]

        # 4. Add credits to user
        db = get_firestore_client()
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            current_credits = 0
            new_credits = credit_amount
            user_ref.set({
                'id': user_id,
                'credits': new_credits,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP,
            })
        else:
            current_credits = user_doc.to_dict().get('credits', 0)
            new_credits = current_credits + credit_amount
            user_ref.update({
                'credits': new_credits,
                'updated_at': firestore.SERVER_TIMESTAMP,
            })

        # 5. Log to credit_history with transaction_id (for duplicate check)
        user_ref.collection('credit_history').add({
            'amount': credit_amount,
            'reason': request.reason,
            'product_id': request.product_id,
            'transaction_id': request.transaction_id,
            'balance_after': new_credits,
            'type': 'purchase',
            'created_at': firestore.SERVER_TIMESTAMP,
        })

        logger.info(f"💰 Added {credit_amount} credits to {user_id} (product: {request.product_id}). {current_credits} -> {new_credits}")

        return AddCreditsResponse(
            success=True,
            credits_before=current_credits,
            credits_after=new_credits,
            amount_added=credit_amount
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error adding credits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/internal/refund")
async def internal_refund_credits(
    request: InternalRefundRequest,
    _: bool = Depends(verify_internal_api_key)
):
    """
    KREDİ İADE - SADECE BACKEND İÇİN
    Bu endpoint sadece internal API key ile çağrılabilir
    Client bu endpoint'e ERİŞEMEZ
    """
    try:
        db = get_firestore_client()
        user_ref = db.collection('users').document(request.user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")

        current_credits = user_doc.to_dict().get('credits', 0)
        new_credits = current_credits + request.amount

        # Update credits
        user_ref.update({
            'credits': new_credits,
            'updated_at': firestore.SERVER_TIMESTAMP,
        })

        # Log to credit_history
        user_ref.collection('credit_history').add({
            'amount': request.amount,
            'reason': f"refund_{request.reason}",
            'job_id': request.job_id,
            'balance_after': new_credits,
            'type': 'refund',
            'created_at': firestore.SERVER_TIMESTAMP,
        })

        logger.info(f"💰 REFUND: Added {request.amount} credits to {request.user_id}. {current_credits} -> {new_credits}")

        return {
            "success": True,
            "credits_before": current_credits,
            "credits_after": new_credits,
            "amount_refunded": request.amount
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error refunding credits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REVENUECAT WEBHOOK (Optional but recommended) ====================

@router.post("/webhook/revenuecat")
async def revenuecat_webhook(
    event: dict,
    authorization: str = Header(None)
):
    """
    RevenueCat Webhook - Otomatik kredi ekleme
    RevenueCat dashboard'da bu URL'yi webhook olarak ayarla
    Bu en güvenli yöntem - client hiç karışmıyor
    """
    # Verify webhook authorization (optional but recommended)
    webhook_auth = os.environ.get("REVENUECAT_WEBHOOK_AUTH", "")
    if webhook_auth and authorization != f"Bearer {webhook_auth}":
        logger.warning("⚠️ Invalid webhook authorization")
        raise HTTPException(status_code=401, detail="Invalid webhook authorization")

    try:
        event_type = event.get("event", {}).get("type")
        app_user_id = event.get("event", {}).get("app_user_id")
        product_id = event.get("event", {}).get("product_id")
        transaction_id = event.get("event", {}).get("transaction_id") or event.get("event", {}).get("id")

        logger.info(f"📥 RevenueCat webhook: {event_type} for user {app_user_id}")

        # Handle purchase events
        if event_type in ["INITIAL_PURCHASE", "NON_RENEWING_PURCHASE", "RENEWAL"]:
            credit_amount = PRODUCT_CREDITS.get(product_id)

            if credit_amount is None:
                logger.warning(f"⚠️ Unknown product in webhook: {product_id}")
                return {"success": True, "message": "Unknown product, skipped"}

            # Check if already processed
            if await check_transaction_already_processed(app_user_id, transaction_id):
                logger.info(f"ℹ️ Transaction already processed: {transaction_id}")
                return {"success": True, "message": "Already processed"}

            # Add credits
            db = get_firestore_client()
            user_ref = db.collection('users').document(app_user_id)
            user_doc = user_ref.get()

            if user_doc.exists:
                current_credits = user_doc.to_dict().get('credits', 0)
                new_credits = current_credits + credit_amount
                user_ref.update({
                    'credits': new_credits,
                    'updated_at': firestore.SERVER_TIMESTAMP,
                })
            else:
                new_credits = credit_amount + 15  # Initial credits + purchase
                user_ref.set({
                    'id': app_user_id,
                    'credits': new_credits,
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'updated_at': firestore.SERVER_TIMESTAMP,
                })
                current_credits = 15

            # Log transaction
            user_ref.collection('credit_history').add({
                'amount': credit_amount,
                'reason': 'webhook_purchase',
                'product_id': product_id,
                'transaction_id': transaction_id,
                'event_type': event_type,
                'balance_after': new_credits,
                'type': 'purchase',
                'created_at': firestore.SERVER_TIMESTAMP,
            })

            logger.info(f"✅ Webhook: Added {credit_amount} credits to {app_user_id}")
            return {"success": True, "credits_added": credit_amount}

        return {"success": True, "message": f"Event {event_type} handled"}

    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        # Return 200 to prevent RevenueCat from retrying
        return {"success": False, "error": str(e)}
