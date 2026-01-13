"""
AIVision - Replicate AI Service with Cost Controls
Handles all AI model calls through Replicate API with verified model IDs.

VERIFIED MODELS - January 2025
All model IDs and version hashes have been tested and verified.
"""
import replicate
import asyncio
import subprocess
import tempfile
import os
import httpx
from typing import Optional, Dict, Any, Tuple
from loguru import logger
from ..config import get_settings, AI_MODELS, COST_LIMITS


class ReplicateService:
    """
    Cost-controlled service for interacting with Replicate AI models.
    All 33 tools with verified model IDs and parameters.
    """

    def __init__(self):
        settings = get_settings()
        self.client = replicate.Client(api_token=settings.REPLICATE_API_TOKEN)
        self.models = AI_MODELS
        self.limits = COST_LIMITS

    # ══════════════════════════════════════════════════════════════════════════
    # CORE REQUEST METHOD
    # ══════════════════════════════════════════════════════════════════════════

    async def _make_request(
        self,
        model_path: str,
        inputs: Dict[str, Any],
        category: str,
        version: Optional[str] = None
    ) -> Tuple[bool, Any]:
        """
        Core method for making Replicate API requests.

        Args:
            model_path: The Replicate model path (e.g., "tencentarc/gfpgan")
            inputs: Input parameters for the model
            category: "video", "magic_edit", or "photo_enhance"
            version: Optional specific model version hash

        Returns:
            Tuple of (success, result_or_error)
        """
        try:
            # Filter None values from inputs
            safe_inputs = {k: v for k, v in inputs.items() if v is not None}

            logger.info(f"[REPLICATE] Running {model_path} ({category})")
            logger.debug(f"Inputs: {safe_inputs}")

            # Build full model identifier with version if provided
            if version:
                model_identifier = f"{model_path}:{version}"
            else:
                model_identifier = model_path

            # Run prediction
            output = await asyncio.to_thread(
                self.client.run,
                model_identifier,
                input=safe_inputs
            )

            return True, self._process_output(output)

        except replicate.exceptions.ReplicateError as e:
            logger.error(f"Replicate API error: {str(e)}")
            return False, f"Replicate API error: {str(e)}"
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return False, str(e)

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO AI TOOLS (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════

    async def generate_video(
        self,
        tool_id: str,
        image_url: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Any]:
        """Generate video using verified video models."""
        model_config = self.models.get(tool_id)
        if not model_config:
            return False, f"Unknown tool: {tool_id}"

        # Special handling for talking_head with TTS pipeline
        if tool_id == "talking_head" and prompt and not kwargs.get("audio_url"):
            # Step 1: Generate audio from text using Chatterbox TTS
            voice_gender = kwargs.get("voice_gender", "female")  # male or female
            logger.info(f"[TTS] Generating {voice_gender} speech from text: {prompt[:50]}...")
            tts_success, tts_result = await self._generate_tts(prompt, model_config, voice_gender)
            if not tts_success:
                return False, f"TTS failed: {tts_result}"

            # Step 2: Use generated audio URL for SadTalker
            audio_url = tts_result.get("url") if isinstance(tts_result, dict) else tts_result
            kwargs["audio_url"] = audio_url
            logger.info(f"[TTS] Audio generated: {audio_url}")

        # Special handling for video_expand: extract last frame + stitch
        if tool_id == "video_expand":
            video_url = kwargs.get("video_url")
            if not video_url:
                return False, "Video Expand requires a video file"

            logger.info(f"[VIDEO_EXPAND] Processing video: {video_url}")
            return await self._process_video_expand(video_url, prompt, model_config)

        inputs = self._prepare_video_inputs(tool_id, image_url, prompt, model_config, kwargs)

        return await self._make_request(
            model_path=model_config["model"],
            inputs=inputs,
            category="video",
            version=model_config.get("version")
        )

    async def _generate_tts(
        self,
        text: str,
        config: Dict[str, Any],
        voice_gender: str = "female"
    ) -> Tuple[bool, Any]:
        """Generate speech from text using Chatterbox TTS."""
        tts_model = config.get("tts_model", "resemble-ai/chatterbox")
        tts_version = config.get("tts_version")

        tts_inputs = {
            "prompt": text,
            "exaggeration": 0.5,  # Neutral emotion
            "cfg_weight": 0.5,
            "temperature": 0.8,
        }

        # Voice cloning: If user provides audio_prompt URL, use it
        # Otherwise Chatterbox uses its default voice
        # TODO: Add male/female voice sample URLs for gender-specific TTS
        # For now, voice_gender parameter is reserved for future use

        return await self._make_request(
            model_path=tts_model,
            inputs=tts_inputs,
            category="tts",
            version=tts_version
        )

    async def _process_video_expand(
        self,
        video_url: str,
        prompt: Optional[str],
        config: Dict[str, Any]
    ) -> Tuple[bool, Any]:
        """
        Process video expansion (lightweight version for Render free tier):
        1. Download input video
        2. Extract last frame using FFmpeg
        3. Call Kling v2.5 with last frame + prompt
        4. Return Kling output directly (no stitching to avoid server crash)

        Note: Stitching removed to prevent Render free tier memory crashes.
        The continuation video is returned directly.
        """
        import base64
        import shutil

        temp_dir = None
        try:
            # Create temp directory for processing
            temp_dir = tempfile.mkdtemp(prefix="video_expand_")
            input_video_path = os.path.join(temp_dir, "input.mp4")
            last_frame_path = os.path.join(temp_dir, "last_frame.jpg")

            # Step 1: Download input video
            logger.info(f"[VIDEO_EXPAND] Step 1: Downloading input video...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(video_url)
                response.raise_for_status()
                with open(input_video_path, "wb") as f:
                    f.write(response.content)
            video_size_mb = len(response.content) / (1024 * 1024)
            logger.info(f"[VIDEO_EXPAND] Input video downloaded: {video_size_mb:.2f} MB")

            # Step 2: Extract last frame using FFmpeg (lightweight operation)
            logger.info(f"[VIDEO_EXPAND] Step 2: Extracting last frame...")
            extract_cmd = [
                "ffmpeg", "-y",
                "-sseof", "-0.1",  # Seek to 0.1 seconds before end
                "-i", input_video_path,
                "-vframes", "1",
                "-q:v", "2",  # High quality JPEG
                last_frame_path
            ]
            result = await asyncio.to_thread(
                subprocess.run, extract_cmd,
                capture_output=True, text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.error(f"[VIDEO_EXPAND] FFmpeg extract error: {result.stderr}")
                return False, f"Failed to extract last frame: {result.stderr}"
            logger.info(f"[VIDEO_EXPAND] Last frame extracted successfully")

            # Step 3: Convert last frame to base64 data URI
            with open(last_frame_path, "rb") as f:
                frame_data = f.read()
            frame_base64 = base64.b64encode(frame_data).decode("utf-8")
            frame_data_uri = f"data:image/jpeg;base64,{frame_base64}"
            logger.info(f"[VIDEO_EXPAND] Frame converted to base64 ({len(frame_data)} bytes)")

            # Step 4: Build prompt for Kling v2.5
            # IMPORTANT: No restrictive phrases! Let the model be creative.
            # Only add quality boosters, never "preserve original", "maintain consistency", etc.
            user_prompt = prompt.strip() if prompt else "continue the scene with smooth motion"
            final_prompt = f"{user_prompt}, high quality, cinematic, 4k, detailed"
            logger.info(f"[VIDEO_EXPAND] Step 3: Calling Kling v2.5 with prompt: '{final_prompt[:100]}...'")

            # Step 5: Call Kling v2.5 with last frame as start_image
            inputs = {
                "prompt": final_prompt,
                "start_image": frame_data_uri,
                "duration": config.get("duration", 5),
                "aspect_ratio": "16:9",
                "negative_prompt": "blurry, low quality, distorted, glitch, static, watermark",
                "guidance_scale": 0.7,
            }

            success, gen_result = await self._make_request(
                model_path=config["model"],
                inputs=inputs,
                category="video",
                version=config.get("version")
            )

            if not success:
                logger.error(f"[VIDEO_EXPAND] Kling generation failed: {gen_result}")
                return False, f"AI generation failed: {gen_result}"

            # Extract video URL from result
            gen_video_url = gen_result.get("url") if isinstance(gen_result, dict) else gen_result
            if not gen_video_url:
                return False, "No video URL in generation result"

            logger.info(f"[VIDEO_EXPAND] SUCCESS! Generated video: {gen_video_url}")

            # Return both URLs - Flutter will stitch them client-side
            return True, {
                "url": gen_video_url,  # Main result (continuation)
                "original_url": video_url,  # Original video for stitching
                "continuation_url": gen_video_url,  # Generated continuation
                "stitch_on_client": True  # Flag for Flutter to stitch
            }

        except subprocess.TimeoutExpired:
            logger.error(f"[VIDEO_EXPAND] FFmpeg timeout")
            return False, "Video processing timed out"
        except httpx.HTTPError as e:
            logger.error(f"[VIDEO_EXPAND] HTTP error: {e}")
            return False, f"Download failed: {str(e)}"
        except Exception as e:
            logger.error(f"[VIDEO_EXPAND] Error: {e}")
            import traceback
            logger.error(f"[VIDEO_EXPAND] Traceback: {traceback.format_exc()}")
            return False, str(e)
        finally:
            # Cleanup temp files immediately to free memory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"[VIDEO_EXPAND] Cleaned up temp dir")
                except Exception as e:
                    logger.warning(f"[VIDEO_EXPAND] Failed to cleanup: {e}")

    def _prepare_video_inputs(
        self,
        tool_id: str,
        image_url: Optional[str],
        prompt: Optional[str],
        config: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare inputs for video generation tools with verified parameters."""
        inputs = {}

        # Get style from kwargs and apply to prompt if present
        style = kwargs.get("style", "")

        def apply_style_to_prompt(base_prompt: str, style_name: str) -> str:
            """Append style to prompt if provided."""
            if style_name and style_name.strip():
                return f"{base_prompt}, {style_name} style"
            return base_prompt

        match tool_id:
            case "ai_hug" | "image_to_video":
                # wan-video/wan-2.2-i2v-fast
                inputs["image"] = image_url
                inputs["prompt"] = prompt or config.get("default_prompt", "smooth cinematic motion")
                inputs["num_frames"] = 81  # Minimum required
                inputs["resolution"] = "480p"
                inputs["frames_per_second"] = 16

            case "video_expand":
                # This is handled specially in generate_video() for frame extraction + stitching
                # The image_url here will be the extracted last frame
                inputs["image"] = image_url
                inputs["prompt"] = prompt or config.get("default_prompt", "Continue the motion smoothly, high quality")
                inputs["num_frames"] = 81  # ~5 seconds at 16fps
                inputs["resolution"] = "480p"
                inputs["frames_per_second"] = 16

            case "text_to_video":
                # bytedance/seedance-1-lite
                base_prompt = prompt or config.get("default_prompt", "cinematic video")
                inputs["prompt"] = apply_style_to_prompt(base_prompt, style)
                # Duration: 2-12 seconds
                duration = kwargs.get("duration", config.get("default_duration", 5))
                inputs["duration"] = max(2, min(12, int(duration)))
                # Resolution: 480p, 720p, 1080p
                resolution = kwargs.get("resolution", config.get("default_resolution", "720p"))
                inputs["resolution"] = resolution if resolution in ["480p", "720p", "1080p"] else "720p"
                # Aspect ratio
                inputs["aspect_ratio"] = kwargs.get("aspect_ratio", config.get("default_aspect_ratio", "16:9"))
                inputs["fps"] = 24

            case "script_to_video":
                # minimax/video-01 - cinematic text-to-video
                base_prompt = prompt or config.get("default_prompt", "cinematic video")
                # Apply style and add aspect ratio hint to prompt
                styled_prompt = apply_style_to_prompt(base_prompt, style)
                aspect_ratio = kwargs.get("aspect_ratio", "9:16")
                # Add cinematic quality modifiers
                final_prompt = f"{styled_prompt}, {aspect_ratio} aspect ratio, high quality, cinematic, professional"
                inputs["prompt"] = final_prompt
                inputs["prompt_optimizer"] = True  # Let minimax enhance the prompt

            case "talking_head":
                # lucataco/sadtalker - talking face animation
                # This is handled specially in generate_video() for TTS pipeline
                audio_url = kwargs.get("audio_url")
                if not image_url:
                    raise ValueError("Talking Head requires a source image (portrait photo)")
                if not audio_url:
                    raise ValueError("Talking Head requires audio. Please record or upload audio, or use AI Voice mode.")

                inputs["source_image"] = image_url
                inputs["driven_audio"] = audio_url
                inputs["preprocess"] = "crop"  # crop, resize, or full
                # enhancer options: "gfpgan", "RestoreFormer" or omit
                if kwargs.get("enhancer"):
                    inputs["enhancer"] = kwargs.get("enhancer")

            case "style_transfer_video":
                # lucataco/animate-diff-vid2vid (verified hash)
                inputs["video_path"] = kwargs.get("video_url")
                inputs["prompt"] = prompt or "artistic style transfer"
                inputs["negative_prompt"] = "blurry, low quality"
                inputs["num_inference_steps"] = 15
                inputs["guidance_scale"] = 7.5

            case "super_slowmo":
                # zsxkib/st-mfnet - ST-MFNet frame interpolation (GAN-based)
                # Supports up to 32x interpolation with better texture preservation
                video_url = kwargs.get("video_url")
                if not video_url:
                    raise ValueError("Super Slow-Mo requires a video input")
                inputs["mp4"] = video_url
                # framerate_multiplier: 2, 4, 8, 16, 32
                speed_factor = int(kwargs.get("speed_factor", kwargs.get("times_to_interpolate", 4)))
                inputs["framerate_multiplier"] = speed_factor
                # CRITICAL: keep_original_duration=False for actual slow motion
                # If True: just smoother video at higher fps (real-time)
                # If False: time is stretched = true slow motion effect
                inputs["keep_original_duration"] = False
                inputs["custom_fps"] = 30  # Output plays at normal 30fps

            case "video_upscale":
                # lucataco/real-esrgan-video (verified hash)
                inputs["video"] = kwargs.get("video_url")
                inputs["scale"] = 2  # Force 2x to save cost

            case "video_bg_remove":
                # arielreplicate/robust_video_matting (verified hash)
                video_url = kwargs.get("video_url")
                if not video_url:
                    raise ValueError("Video BG Remove requires a video input")
                inputs["input_video"] = video_url
                # Output format: green_screen or transparent
                output_format = kwargs.get("output_format", "green_screen")
                if output_format == "green_screen":
                    # Green background for mobile editors
                    inputs["background_color"] = "green"
                else:
                    # Transparent/alpha channel (WebM with alpha)
                    inputs["background_color"] = "transparent"

            case "face_swap_video":
                # arabyai-replicate/roop_face_swap - Roop face swap
                # swap_image = the face image (face to use)
                # target_video = the video to swap face into
                source_face = kwargs.get("face_image_url") or image_url
                target_video = kwargs.get("video_url")

                if not source_face:
                    raise ValueError("Face Swap Video requires a source face image")
                if not target_video:
                    raise ValueError("Face Swap Video requires a target video")

                inputs["swap_image"] = source_face
                inputs["target_video"] = target_video

        return inputs

    # ══════════════════════════════════════════════════════════════════════════
    # PHOTO ENHANCE TOOLS (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════

    async def enhance_photo(
        self,
        tool_id: str,
        image_url: str,
        prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Any]:
        """Enhance photo using verified photo enhancement models."""
        model_config = self.models.get(tool_id)
        if not model_config:
            return False, f"Unknown tool: {tool_id}"

        inputs = self._prepare_enhance_inputs(tool_id, image_url, prompt, model_config, kwargs)

        return await self._make_request(
            model_path=model_config["model"],
            inputs=inputs,
            category="photo_enhance",
            version=model_config.get("version")
        )

    def _prepare_enhance_inputs(
        self,
        tool_id: str,
        image_url: str,
        prompt: Optional[str],
        config: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare inputs for photo enhancement tools with verified parameters."""
        inputs = {}

        # Get style from kwargs and apply to prompt if present
        style = kwargs.get("style", "")

        def apply_style_to_prompt(base_prompt: str, style_name: str) -> str:
            """Append style to prompt if provided."""
            if style_name and style_name.strip():
                return f"{base_prompt}, {style_name} style"
            return base_prompt

        match tool_id:
            case "face_clarify":
                # tencentarc/gfpgan (verified hash)
                inputs["img"] = image_url
                inputs["version"] = "v1.4"
                inputs["scale"] = 2

            case "old_photo_restore" | "scratch_remover":
                # microsoft/bringing-old-photos-back-to-life (verified hash)
                inputs["image"] = image_url
                inputs["HR"] = False  # Use low res for speed
                inputs["with_scratch"] = (tool_id == "scratch_remover")

            case "colorize":
                # arielreplicate/deoldify_image (verified hash)
                inputs["input_image"] = image_url
                inputs["model_name"] = "Artistic"  # Required: "Artistic" or "Stable"
                inputs["render_factor"] = 35

            case "4k_upscale":
                # cjwbw/supir-v0q - SUPIR image upscaler
                inputs["image"] = image_url
                inputs["upscale"] = kwargs.get("scale", 2)
                inputs["min_size"] = 1024
                inputs["edm_steps"] = 30  # Balance quality/speed
                inputs["s_cfg"] = 7.5
                inputs["color_fix_type"] = "Wavelet"

            case "relight":
                # zsxkib/ic-light - portrait relighting with text guidance
                inputs["subject_image"] = image_url
                inputs["prompt"] = prompt or config.get("default_prompt", "soft natural lighting, professional photo")
                # light_source options: "Left Light", "Right Light", "Top Light", "Bottom Light" or omit
                light_source = kwargs.get("light_source")
                if light_source and light_source != "None":
                    inputs["light_source"] = light_source
                inputs["steps"] = 25
                inputs["cfg"] = 2.0
                inputs["highres_scale"] = 1.5
                inputs["lowres_denoise"] = 0.9
                inputs["highres_denoise"] = 0.5

            case "low_light_fix":
                # megvii-research/nafnet - image enhancement
                inputs["image"] = image_url
                # Use deblurring for better low-light results
                inputs["task_type"] = "Image Deblurring (REDS)"

            case "denoise":
                # sczhou/codeformer (verified hash)
                inputs["image"] = image_url
                inputs["upscale"] = 1
                inputs["face_upsample"] = True
                inputs["background_enhance"] = True
                inputs["codeformer_fidelity"] = 0.7

            case "anime_yourself":
                # asiryan/meina-mix-v11 - Anime style transformation
                inputs["image"] = image_url
                base_prompt = prompt or config.get("default_prompt", "anime style portrait, beautiful detailed anime art, vibrant colors, masterpiece")
                inputs["prompt"] = apply_style_to_prompt(base_prompt, style)
                inputs["negative_prompt"] = "ugly, blurry, low quality, deformed, disfigured"
                inputs["strength"] = 0.65  # Balance between original and anime
                inputs["num_inference_steps"] = 30
                inputs["guidance_scale"] = 7.5

            case "portrait_mode":
                # lucataco/remove-bg (verified hash)
                # Note: Client-side will add blur effect
                inputs["image"] = image_url

            case "color_correct":
                # google-research/maxim (verified hash)
                inputs["image"] = image_url
                inputs["model"] = "Image Enhancement"

        return inputs

    # ══════════════════════════════════════════════════════════════════════════
    # MAGIC EDIT TOOLS (11 Tools)
    # ══════════════════════════════════════════════════════════════════════════

    async def edit_image(
        self,
        tool_id: str,
        image_url: str,
        prompt: Optional[str] = None,
        mask_url: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Any]:
        """Edit image using verified magic edit models."""
        model_config = self.models.get(tool_id)
        if not model_config:
            return False, f"Unknown tool: {tool_id}"

        inputs = self._prepare_edit_inputs(tool_id, image_url, prompt, mask_url, model_config, kwargs)

        return await self._make_request(
            model_path=model_config["model"],
            inputs=inputs,
            category="magic_edit",
            version=model_config.get("version")
        )

    def _prepare_edit_inputs(
        self,
        tool_id: str,
        image_url: str,
        prompt: Optional[str],
        mask_url: Optional[str],
        config: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare inputs for magic edit tools with verified parameters."""
        inputs = {}

        # Get style from kwargs and apply to prompt if present
        style = kwargs.get("style", "")

        def apply_style_to_prompt(base_prompt: str, style_name: str) -> str:
            """Append style to prompt if provided."""
            if style_name and style_name.strip():
                return f"{base_prompt}, {style_name} style"
            return base_prompt

        match tool_id:
            case "magic_eraser":
                # bria/eraser - SOTA object removal
                # NOTE: This model REQUIRES a mask image
                inputs["image"] = image_url
                if mask_url:
                    inputs["mask"] = mask_url
                else:
                    raise ValueError("Magic Eraser requires a mask. Please draw on the area you want to remove.")

            case "ai_headshot":
                # tencentarc/photomaker (verified hash)
                inputs["input_image"] = image_url
                base_prompt = prompt or config.get("default_prompt")
                inputs["prompt"] = apply_style_to_prompt(base_prompt, style)
                # Map style to photomaker style_name if applicable
                style_mapping = {
                    "cinematic": "Cinematic",
                    "anime": "Comic book",
                    "digital art": "Digital Art",
                    "fantasy": "Fantasy art",
                    "neon": "Neon punk",
                    "lowpoly": "Lowpoly",
                }
                inputs["style_name"] = style_mapping.get(style.lower(), "Photographic (Default)") if style else "Photographic (Default)"
                inputs["num_steps"] = 20
                inputs["guidance_scale"] = 5
                inputs["style_strength_ratio"] = 20

            case "clothes_swap":
                # cuuupid/idm-vton (verified hash)
                inputs["human_img"] = image_url
                inputs["garm_img"] = kwargs.get("garment_image_url")
                inputs["garment_des"] = prompt or "clothing item"
                inputs["is_checked"] = True
                inputs["is_checked_crop"] = False
                inputs["denoise_steps"] = 30

            case "bg_remix" | "outpainting" | "sky_replace" | "tattoo_tryon":
                # black-forest-labs/flux-fill-schnell (inpainting)
                inputs["image"] = image_url
                inputs["mask"] = mask_url
                base_prompt = prompt or config.get("default_prompt", "seamless natural blend")
                inputs["prompt"] = apply_style_to_prompt(base_prompt, style)
                inputs["num_inference_steps"] = 4
                inputs["guidance_scale"] = 3.5

            case "sticker_maker":
                # lucataco/remove-bg (verified hash)
                # Note: Client-side will add outline effect
                inputs["image"] = image_url

            case "interior_design":
                # adirik/interior-design - room redesign AI
                inputs["image"] = image_url
                base_prompt = prompt or config.get("default_prompt", "modern interior design, stylish furniture")
                inputs["prompt"] = apply_style_to_prompt(base_prompt, style)
                inputs["negative_prompt"] = "lowres, watermark, banner, logo, text, deformed, blurry, blur, out of focus, ugly"
                inputs["num_inference_steps"] = 50
                inputs["guidance_scale"] = 15
                inputs["prompt_strength"] = 0.8

            case "product_shoot":
                # zsxkib/ic-light (same as relight, for product photography)
                # NOTE: ic-light requires "subject_image" not "image"
                inputs["subject_image"] = image_url
                base_prompt = prompt or config.get("default_prompt")
                inputs["prompt"] = apply_style_to_prompt(base_prompt, style)
                inputs["light_source"] = "Top Light"
                inputs["num_inference_steps"] = 25

            case "text_effects":
                # jagilley/controlnet-canny (verified hash)
                inputs["image"] = image_url
                base_prompt = prompt or "stylized text with effects"
                inputs["prompt"] = apply_style_to_prompt(base_prompt, style)
                inputs["num_samples"] = "1"
                inputs["image_resolution"] = "512"
                inputs["ddim_steps"] = 20
                inputs["scale"] = 9

            case "style_transfer":
                # fofr/style-transfer - Artistic style transfer with face preservation
                style_url = kwargs.get("style_url")
                if not style_url:
                    raise ValueError("Style Transfer requires a style reference image")
                # CORRECT PARAMETER NAMES for fofr/style-transfer:
                inputs["structure_image"] = image_url  # User's photo (source face)
                inputs["style_image"] = style_url      # Style reference (artistic style)
                # BALANCED: 0.65 = apply style strongly while keeping face recognizable
                inputs["structure_denoising_strength"] = config.get("structure_denoising_strength", 0.65)
                # Preserve face structure but allow artistic interpretation
                inputs["structure_depth_strength"] = config.get("structure_depth_strength", 0.9)
                # Style strength - how much to apply the style image
                inputs["style_strength"] = 0.8  # Strong style application
                # Prompt: emphasize artistic transformation with face preservation
                inputs["prompt"] = "artistic style transfer, beautiful artwork, same person, recognizable face, high quality masterpiece, detailed artistic rendering"
                # Prevent bad outputs
                inputs["negative_prompt"] = "deformed face, wrong face, different person, blurry, ugly, bad anatomy, extra limbs"

        return inputs

    # ══════════════════════════════════════════════════════════════════════════
    # UNIFIED RUN METHOD
    # ══════════════════════════════════════════════════════════════════════════

    async def run_prediction(
        self,
        tool_id: str,
        input_params: Dict[str, Any]
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Run a prediction for any tool.
        Routes to the appropriate handler based on tool category.
        """
        model_config = self.models.get(tool_id)
        if not model_config:
            return False, f"Unknown tool: {tool_id}", None

        category = model_config.get("category", "photo_enhance")

        # Extract common params
        image_url = input_params.get("image_url")
        prompt = input_params.get("prompt")
        mask_url = input_params.get("mask_url")

        # Filter out known params for kwargs
        extra_params = {k: v for k, v in input_params.items()
                       if k not in ["image_url", "prompt", "mask_url", "tool_id"]}

        # Route to appropriate handler
        if category == "video":
            success, result = await self.generate_video(
                tool_id=tool_id,
                image_url=image_url,
                prompt=prompt,
                **extra_params
            )
        elif category == "magic_edit":
            success, result = await self.edit_image(
                tool_id=tool_id,
                image_url=image_url,
                prompt=prompt,
                mask_url=mask_url,
                **extra_params
            )
        else:  # photo_enhance
            success, result = await self.enhance_photo(
                tool_id=tool_id,
                image_url=image_url,
                prompt=prompt,
                **extra_params
            )

        # Add client-side processing flag if needed
        client_processing = model_config.get("client_side_processing")

        return success, result, client_processing

    # ══════════════════════════════════════════════════════════════════════════
    # ASYNC PREDICTION METHODS (for long-running tasks)
    # ══════════════════════════════════════════════════════════════════════════

    async def start_prediction_async(
        self,
        tool_id: str,
        input_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Start an async prediction. Returns prediction ID for polling."""
        try:
            model_config = self.models.get(tool_id)
            if not model_config:
                return False, None, f"Unknown tool: {tool_id}"

            category = model_config.get("category", "photo_enhance")

            # Prepare inputs based on category
            if category == "video":
                inputs = self._prepare_video_inputs(
                    tool_id,
                    input_params.get("image_url"),
                    input_params.get("prompt"),
                    model_config,
                    input_params
                )
            elif category == "magic_edit":
                inputs = self._prepare_edit_inputs(
                    tool_id,
                    input_params.get("image_url"),
                    input_params.get("prompt"),
                    input_params.get("mask_url"),
                    model_config,
                    input_params
                )
            else:
                inputs = self._prepare_enhance_inputs(
                    tool_id,
                    input_params.get("image_url"),
                    input_params.get("prompt"),
                    model_config,
                    input_params
                )

            # Filter None values
            safe_inputs = {k: v for k, v in inputs.items() if v is not None}

            logger.info(f"[ASYNC] Starting prediction for {tool_id}")

            # Build model identifier
            model_path = model_config["model"]
            version = model_config.get("version")

            # Create prediction - use version if available, otherwise get latest version
            if version:
                prediction = await asyncio.to_thread(
                    self.client.predictions.create,
                    version=version,
                    input=safe_inputs
                )
            else:
                # Get the latest version of the model first
                try:
                    model = await asyncio.to_thread(
                        self.client.models.get,
                        model_path
                    )
                    latest_version = model.latest_version
                    if latest_version:
                        logger.info(f"[ASYNC] Using latest version: {latest_version.id}")
                        prediction = await asyncio.to_thread(
                            self.client.predictions.create,
                            version=latest_version.id,
                            input=safe_inputs
                        )
                    else:
                        # Fallback to model parameter
                        prediction = await asyncio.to_thread(
                            self.client.predictions.create,
                            model=model_path,
                            input=safe_inputs
                        )
                except Exception as e:
                    logger.warning(f"Failed to get latest version, trying model param: {e}")
                    prediction = await asyncio.to_thread(
                        self.client.predictions.create,
                        model=model_path,
                        input=safe_inputs
                    )

            return True, prediction.id, None

        except Exception as e:
            logger.error(f"Async prediction error: {str(e)}")
            return False, None, str(e)

    async def get_prediction_status(self, prediction_id: str) -> Dict[str, Any]:
        """Get status of async prediction."""
        try:
            prediction = await asyncio.to_thread(
                self.client.predictions.get,
                prediction_id
            )

            return {
                "id": prediction.id,
                "status": prediction.status,
                "output": prediction.output,
                "error": prediction.error,
                "logs": prediction.logs,
                "metrics": prediction.metrics
            }
        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return {"status": "error", "error": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def _process_output(self, output: Any) -> Dict[str, Any]:
        """Process and normalize model output."""
        logger.info(f"[REPLICATE] Raw output type: {type(output)}")
        logger.info(f"[REPLICATE] Raw output: {output}")

        # Helper to extract URL from FileOutput or similar objects
        def extract_url(item: Any) -> str:
            # FileOutput objects have a direct string representation that is the URL
            if hasattr(item, '__str__'):
                url = str(item)
                # Verify it looks like a URL
                if url.startswith('http'):
                    return url
            # Try .url attribute
            if hasattr(item, 'url'):
                return str(item.url)
            # Last resort
            return str(item)

        # If it's a generator, consume it first
        if hasattr(output, '__next__'):
            logger.info("[REPLICATE] Output is a generator, consuming...")
            output = list(output)
            logger.info(f"[REPLICATE] Generator consumed: {output}")

        # Handle FileOutput directly (Replicate's file output wrapper)
        type_name = type(output).__name__
        if type_name == 'FileOutput' or 'FileOutput' in str(type(output)):
            url = extract_url(output)
            logger.info(f"[REPLICATE] FileOutput detected, extracted URL: {url}")
            return {"url": url, "type": "single"}

        if isinstance(output, str):
            logger.info(f"[REPLICATE] Returning single URL string: {output}")
            return {"url": output, "type": "single"}
        elif isinstance(output, list):
            if len(output) == 0:
                logger.warning("[REPLICATE] Output list is empty!")
                return {"url": None, "type": "empty"}
            # Extract URLs from list items (may be FileOutput objects)
            urls = [extract_url(item) for item in output]
            if len(urls) == 1:
                logger.info(f"[REPLICATE] Returning single URL from list: {urls[0]}")
                return {"url": urls[0], "type": "single"}
            logger.info(f"[REPLICATE] Returning multiple URLs: {urls}")
            return {"urls": urls, "type": "multiple"}
        elif hasattr(output, 'url'):
            url = str(output.url)
            logger.info(f"[REPLICATE] Returning URL from object: {url}")
            return {"url": url, "type": "single"}
        elif hasattr(output, '__iter__'):
            items = list(output)
            urls = [extract_url(item) for item in items]
            logger.info(f"[REPLICATE] Iterable converted to list: {urls}")
            if len(urls) == 0:
                logger.warning("[REPLICATE] Iterable was empty!")
                return {"url": None, "type": "empty"}
            if len(urls) == 1:
                return {"url": urls[0], "type": "single"}
            return {"urls": urls, "type": "multiple"}
        else:
            # Try to extract URL anyway
            url = extract_url(output)
            if url.startswith('http'):
                logger.info(f"[REPLICATE] Extracted URL from unknown type: {url}")
                return {"url": url, "type": "single"}
            logger.info(f"[REPLICATE] Unknown output type, stringified: {url}")
            return {"result": url, "type": "unknown"}

    def get_estimated_time(self, tool_id: str) -> int:
        """Get estimated processing time in seconds."""
        time_estimates = {
            # VIDEO AI
            "ai_hug": 45, "image_to_video": 45, "text_to_video": 60,
            "talking_head": 30, "video_expand": 45, "style_transfer_video": 40,
            "super_slowmo": 20, "video_upscale": 60, "video_bg_remove": 40,
            "face_swap_video": 45, "script_to_video": 60,
            # PHOTO ENHANCE
            "face_clarify": 8, "old_photo_restore": 20, "colorize": 15,
            "4k_upscale": 15, "relight": 15, "low_light_fix": 8,
            "scratch_remover": 20, "denoise": 10, "anime_yourself": 12,
            "portrait_mode": 8, "color_correct": 10,
            # MAGIC EDIT
            "magic_eraser": 8, "ai_headshot": 15, "clothes_swap": 25,
            "bg_remix": 10, "sticker_maker": 5, "outpainting": 10,
            "sky_replace": 10, "interior_design": 20, "product_shoot": 15,
            "text_effects": 10, "tattoo_tryon": 10, "style_transfer": 15,
        }
        return time_estimates.get(tool_id, 15)

    def get_credit_cost(self, tool_id: str) -> int:
        """Get credit cost for tools (for user-facing display)."""
        costs = {
            # VIDEO AI
            "ai_hug": 8, "image_to_video": 8, "text_to_video": 10,
            "talking_head": 6, "video_expand": 8, "style_transfer_video": 8,
            "super_slowmo": 4, "video_upscale": 8, "video_bg_remove": 6,
            "face_swap_video": 8, "script_to_video": 10,
            # PHOTO ENHANCE
            "face_clarify": 2, "old_photo_restore": 3, "colorize": 2,
            "4k_upscale": 3, "relight": 3, "low_light_fix": 2,
            "scratch_remover": 3, "denoise": 2, "anime_yourself": 2,
            "portrait_mode": 2, "color_correct": 2,
            # MAGIC EDIT
            "magic_eraser": 3, "ai_headshot": 5, "clothes_swap": 5,
            "bg_remix": 3, "sticker_maker": 2, "outpainting": 3,
            "sky_replace": 3, "interior_design": 4, "product_shoot": 3,
            "text_effects": 3, "tattoo_tryon": 3, "style_transfer": 4,
        }
        return costs.get(tool_id, 2)

    def get_model_info(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get model info for a tool."""
        return self.models.get(tool_id)


# Global service instance
replicate_service = ReplicateService()
