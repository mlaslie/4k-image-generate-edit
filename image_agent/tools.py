import io
import os
import gc
import time
import json
import base64
import asyncio
import logging
from typing import Optional, Tuple, Any, Dict
from PIL import Image, PngImagePlugin
from google import genai
from google.genai import types
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

# Primary model requested
MODEL_NAME = "gemini-3-pro-image"
GCS_BUCKET_NAME = os.environ.get("GCS_ARTIFACT_BUCKET", "image-editing-agent-artifacts")

# --- Client & Logger Singletons ---
_gcp_logging_client: Optional[Any] = None
_gcp_audit_logger: Optional[Any] = None
_gcp_storage_client: Optional[Any] = None
_genai_client: Optional[genai.Client] = None
_logging_init_attempted: bool = False


def get_logging_client():
    """Initializes and caches a singleton Google Cloud Logging client & logger."""
    global _gcp_logging_client, _gcp_audit_logger, _logging_init_attempted
    if _logging_init_attempted:
        return _gcp_logging_client, _gcp_audit_logger

    _logging_init_attempted = True
    use_enterprise = (
        os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "1") == "1"
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "1") == "1"
    )
    if not use_enterprise:
        return None, None

    try:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "laslie-demo-project")
        import google.cloud.logging

        _gcp_logging_client = google.cloud.logging.Client(project=project)
        _gcp_audit_logger = _gcp_logging_client.logger("image_agent_user_audit")
    except Exception as e:
        logger.debug("Cloud Logging initialization bypassed: %s", e)
    return _gcp_logging_client, _gcp_audit_logger


def get_storage_client():
    """Initializes and caches a singleton Google Cloud Storage client."""
    global _gcp_storage_client
    if _gcp_storage_client is None:
        try:
            import google.cloud.storage

            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "laslie-demo-project")
            _gcp_storage_client = google.cloud.storage.Client(project=project)
        except Exception as e:
            logger.debug("Cloud Storage client init bypassed: %s", e)
    return _gcp_storage_client


def get_genai_client() -> genai.Client:
    """Initializes and caches a singleton Google GenAI client configured for ADC / Vertex AI."""
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT", "laslie-demo-project")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        use_enterprise = (
            os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "1") == "1"
            or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "1") == "1"
        )

        if api_key:
            _genai_client = genai.Client(api_key=api_key)
        elif use_enterprise and project:
            _genai_client = genai.Client(vertexai=True, project=project, location=location)
        else:
            _genai_client = genai.Client()
    return _genai_client


def get_user_identity(tool_context: Optional[ToolContext] = None) -> str:
    """
    Extracts the user identity (email / ID) from Gemini Enterprise OIDC tokens,
    session state, tool_context.user_id, or metadata.
    Returns 'unknown' if not resolvable without failing.
    """
    if not tool_context:
        return "unknown"

    try:
        # 1. Check OIDC auth response / token claims if available
        if hasattr(tool_context, "get_auth_response"):
            try:
                auth_resp = tool_context.get_auth_response()
                if auth_resp and isinstance(auth_resp, dict):
                    for key in ["email", "user_email", "preferred_username", "sub"]:
                        val = auth_resp.get(key)
                        if val and isinstance(val, str) and val.strip():
                            return val.strip()
            except Exception:
                pass

        # 2. Check session state or invocation state for OIDC / user claims
        state = getattr(tool_context, "state", {}) or {}
        if isinstance(state, dict):
            for key in ["user_email", "email", "user_id", "user", "preferred_username"]:
                val = state.get(key)
                if val and isinstance(val, str) and val.strip():
                    return val.strip()

        # 3. Direct tool_context.user_id property
        user_id = getattr(tool_context, "user_id", None)
        if user_id and isinstance(user_id, str) and user_id.strip():
            return user_id.strip()

        # 4. Check session.user_id
        if hasattr(tool_context, "session") and tool_context.session:
            session_user_id = getattr(tool_context.session, "user_id", None)
            if session_user_id and isinstance(session_user_id, str) and session_user_id.strip():
                return session_user_id.strip()

        # 5. Check custom_metadata
        if hasattr(tool_context, "custom_metadata") and tool_context.custom_metadata:
            for key in ["user_email", "email", "user_id", "user", "principal"]:
                val = tool_context.custom_metadata.get(key)
                if val and isinstance(val, str) and val.strip():
                    return val.strip()
    except Exception:
        pass

    return "unknown"


def _write_cloud_log_struct(log_payload: Dict[str, Any]):
    """Helper to write log entry with ReasoningEngine resource attachment in worker thread."""
    try:
        _, gcp_logger = get_logging_client()
        if gcp_logger:
            from google.cloud.logging_v2.resource import Resource

            engine_id = os.environ.get("REASONING_ENGINE_ID", "1371774346313334784")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "laslie-demo-project")

            resource = Resource(
                type="aiplatform.googleapis.com/ReasoningEngine",
                labels={
                    "reasoning_engine_id": engine_id,
                    "location": location,
                    "resource_container": f"projects/{project}",
                },
            )
            gcp_logger.log_struct(log_payload, resource=resource, severity="INFO")
    except Exception as e:
        logger.debug("Cloud Logging struct write exception: %s", e)


def log_user_activity(action: str, user_id: str, details: Dict[str, Any]):
    """
    Logs user activity to standard logging and Google Cloud Logging via singleton and ReasoningEngine resource.
    Guaranteed never to raise exceptions or interrupt tool execution.
    """
    try:
        safe_user_id = user_id if (user_id and isinstance(user_id, str) and user_id.strip()) else "unknown"
        log_payload = {
            "event": "image_agent_user_activity",
            "action": action,
            "user_id": safe_user_id,
            "timestamp": time.time(),
            **details,
        }

        logger.info(
            "User [%s] performed action '%s' - Details: %s",
            safe_user_id,
            action,
            json.dumps(log_payload, default=str),
        )

        # Offload cloud logging write to thread pool if in active event loop
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _write_cloud_log_struct, log_payload)
        except RuntimeError:
            _write_cloud_log_struct(log_payload)
    except Exception as e:
        logger.debug("log_user_activity outer handler exception: %s", e)


def calculate_dpi_for_resolution(width: int, height: int) -> Tuple[int, int]:
    """
    Determines optimal DPI metadata based on image resolution.
    - 4K / Ultra-HD (>= 3840px): 300 DPI (High-quality print standard)
    - 2K / Quad-HD (>= 2048px): 240 DPI
    - 1K / Full-HD (>= 1024px): 150 DPI
    - Standard Web (< 1024px): 72 DPI
    """
    max_dimension = max(width, height)
    if max_dimension >= 3840:
        return (300, 300)
    elif max_dimension >= 2048:
        return (240, 240)
    elif max_dimension >= 1024:
        return (150, 150)
    else:
        return (72, 72)


def _sync_process_and_apply_dpi(
    image_bytes: bytes,
) -> Tuple[bytes, str, Tuple[int, int], Tuple[int, int]]:
    """Synchronous CPU worker for DPI metadata injection."""
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    dpi = calculate_dpi_for_resolution(width, height)

    mime_type = "image/png"
    output_buf = io.BytesIO()
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("Resolution", f"{width}x{height}")
    png_info.add_text("DPI", f"{dpi[0]}")
    img.save(output_buf, format="PNG", dpi=dpi, pnginfo=png_info)

    return output_buf.getvalue(), mime_type, (width, height), dpi


async def process_and_apply_dpi(
    image_bytes: bytes,
) -> Tuple[bytes, str, Tuple[int, int], Tuple[int, int]]:
    """
    Asynchronously injects DPI metadata in a worker thread.
    Prevents blocking the async event loop.
    """
    return await asyncio.to_thread(_sync_process_and_apply_dpi, image_bytes)


def _extract_image_bytes(response: Any) -> Optional[bytes]:
    """Helper to extract raw image bytes from a GenerateContent response."""
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if part.inline_data and part.inline_data.data:
                        return part.inline_data.data
    return None


async def _upload_to_gcs_async(filename: str, data: bytes, mime_type: str) -> str:
    """Uploads raw image bytes to GCS bucket in worker thread and returns gs:// URI."""
    def _sync_upload():
        try:
            client = get_storage_client()
            if client:
                bucket = client.bucket(GCS_BUCKET_NAME)
                blob = bucket.blob(f"artifacts/{filename}")
                blob.upload_from_string(data, content_type=mime_type)
        except Exception as e:
            logger.warning("GCS direct upload error: %s", e)
        return f"gs://{GCS_BUCKET_NAME}/artifacts/{filename}"

    return await asyncio.to_thread(_sync_upload)


async def _execute_gemini_3_pro_image(
    prompt: str,
    input_image_uri: Optional[str] = None,
    input_image_bytes: Optional[bytes] = None,
    input_mime_type: str = "image/png",
    aspect_ratio: str = "",
    image_size: str = "4K",
    max_retries: int = 3,
) -> bytes:
    """
    Executes gemini-3-pro-image on Vertex AI using native generation_config image_config.
    Accepts zero-memory GCS URIs (types.Part.from_uri) to eliminate binary payload serialization.
    """
    client = get_genai_client()
    contents = []

    if input_image_uri:
        # Pass GCS URI directly without loading raw bytes into container memory
        contents.append(types.Part.from_uri(file_uri=input_image_uri, mime_type=input_mime_type))
    elif input_image_bytes:
        contents.append(types.Part.from_bytes(data=input_image_bytes, mime_type=input_mime_type))

    contents.append(prompt)

    # Build image_config cleanly as generation_config parameters
    image_config_kwargs: Dict[str, Any] = {
        "image_size": image_size or "4K",
    }
    if aspect_ratio and aspect_ratio.strip():
        image_config_kwargs["aspect_ratio"] = aspect_ratio.strip()

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(**image_config_kwargs),
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            raw_bytes = _extract_image_bytes(response)
            if raw_bytes:
                return raw_bytes
        except Exception as e:
            last_err = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (attempt + 1) * 3
                logger.warning("Rate limit 429 hit. Retrying in %ss...", wait_time)
                await asyncio.sleep(wait_time)
            else:
                raise

    raise last_err if last_err else RuntimeError(f"No image was returned by {MODEL_NAME}.")


async def generate_4k_image(
    prompt: str,
    aspect_ratio: str = "",
    image_size: str = "4K",
    negative_prompt: str = "",
    tool_context: Optional[ToolContext] = None,
) -> str:
    """
    Generates a high-quality image using gemini-3-pro-image with native generation_config parameters.
    Saves artifacts as zero-memory file_data references (gs://) in context.

    Args:
        prompt: Detailed description of the image to generate.
        aspect_ratio: Aspect ratio of the output image (e.g., '1:1', '16:9', '9:16', '4:3', '3:4', '21:9', '3:2', '2:3').
                      If omitted or empty, the model automatically determines the best composition aspect ratio.
        image_size: Target resolution ('4K', '2K', '1K'). Defaults to '4K'.
        negative_prompt: Optional elements to exclude from the generated image.
        tool_context: ADK ToolContext for artifact storage and session management.

    Returns:
        A message detailing the generated image, resolution, DPI metadata, GCS URI, and artifact filename.
    """
    user_id = get_user_identity(tool_context)
    log_user_activity(
        action="generate_image_request",
        user_id=user_id,
        details={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio or "auto",
            "image_size": image_size or "4K",
            "negative_prompt": negative_prompt,
        },
    )

    try:
        full_prompt = f"{prompt} (Exclude: {negative_prompt})" if negative_prompt else prompt
        raw_bytes = await _execute_gemini_3_pro_image(
            prompt=full_prompt,
            aspect_ratio=aspect_ratio,
            image_size=image_size or "4K",
        )

        # Inject DPI metadata asynchronously in worker thread
        processed_bytes, mime_type, (width, height), (dpi_x, dpi_y) = await process_and_apply_dpi(raw_bytes)

        timestamp = int(time.time())
        filename = f"generated_{timestamp}.png"

        # Stream directly to GCS external storage
        gcs_uri = await _upload_to_gcs_async(filename, processed_bytes, mime_type)

        # Save artifact as lightweight file_data reference (ZERO raw bytes in memory context)
        if tool_context:
            artifact_file_ref = types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type)
            metadata = {
                "model": MODEL_NAME,
                "resolution": f"{width}x{height}",
                "dpi": f"{dpi_x}x{dpi_y}",
                "aspect_ratio": aspect_ratio or "auto",
                "image_size": image_size or "4K",
                "prompt": prompt,
                "user_id": user_id,
                "gcs_uri": gcs_uri,
            }
            await tool_context.save_artifact(filename=filename, artifact=artifact_file_ref, custom_metadata=metadata)

        # Immediately free container memory buffers
        del raw_bytes, processed_bytes
        gc.collect()

        log_user_activity(
            action="generate_image_success",
            user_id=user_id,
            details={
                "prompt": prompt,
                "artifact_filename": filename,
                "gcs_uri": gcs_uri,
                "resolution": f"{width}x{height}",
                "dpi": f"{dpi_x}x{dpi_y}",
                "aspect_ratio": aspect_ratio or "auto",
                "image_size": image_size or "4K",
            },
        )

        aspect_label = aspect_ratio if aspect_ratio else "Auto-determined by model"
        return (
            f"Successfully generated image with {MODEL_NAME}:\n"
            f"- **Filename / Artifact**: `{filename}`\n"
            f"- **GCS Storage**: `{gcs_uri}`\n"
            f"- **Model**: `{MODEL_NAME}`\n"
            f"- **Resolution**: {width}x{height} ({image_size or '4K'})\n"
            f"- **DPI Metadata**: {dpi_x} DPI\n"
            f"- **Aspect Ratio**: {aspect_label}\n"
            f"- **Prompt**: \"{prompt}\""
        )
    except Exception as e:
        logger.exception("Error generating image with %s for user %s", MODEL_NAME, user_id)
        log_user_activity(
            action="generate_image_failure",
            user_id=user_id,
            details={"prompt": prompt, "error": str(e)},
        )
        return f"Failed to generate image with {MODEL_NAME}: {str(e)}"


async def edit_4k_image(
    prompt: str,
    image_artifact_name: str = "",
    aspect_ratio: str = "",
    image_size: str = "4K",
    tool_context: Optional[ToolContext] = None,
) -> str:
    """
    Modifies or edits an existing image using text instructions with gemini-3-pro-image.
    Uses zero-memory GCS file_data references (gs://) to eliminate multi-turn memory accumulation.

    Args:
        prompt: Instructions for modifying or transforming the image.
        image_artifact_name: The filename/artifact name or gs:// URI of the base image to modify.
                            If omitted, the most recently saved artifact or user attachment will be used.
        aspect_ratio: Desired output aspect ratio (e.g., '1:1', '16:9', '9:16', '4:3', '3:4', '21:9', '3:2', '2:3').
                      If omitted or empty, preserves / auto-determines aspect ratio.
        image_size: Target resolution ('4K', '2K', '1K'). Defaults to '4K'.
        tool_context: ADK ToolContext for artifact retrieval and storage.

    Returns:
        A message detailing the edited image, resolution, DPI metadata, GCS URI, and new artifact filename.
    """
    user_id = get_user_identity(tool_context)
    log_user_activity(
        action="edit_image_request",
        user_id=user_id,
        details={
            "prompt": prompt,
            "image_artifact_name": image_artifact_name,
            "aspect_ratio": aspect_ratio or "auto",
            "image_size": image_size or "4K",
        },
    )

    try:
        input_image_uri: Optional[str] = None
        input_image_bytes: Optional[bytes] = None
        input_mime_type: str = "image/png"

        # 1. Direct gs:// URI given in parameter
        if image_artifact_name and image_artifact_name.startswith("gs://"):
            input_image_uri = image_artifact_name

        # 2. Check artifact store in tool_context
        if not input_image_uri and tool_context:
            target_name = image_artifact_name
            if not target_name:
                available_artifacts = await tool_context.list_artifacts()
                if available_artifacts:
                    target_name = available_artifacts[-1]

            if target_name:
                part = await tool_context.load_artifact(target_name)
                if part:
                    # Check for file_data reference first (preferred: 0 bytes in memory)
                    if hasattr(part, "file_data") and part.file_data and part.file_data.file_uri:
                        input_image_uri = part.file_data.file_uri
                        input_mime_type = part.file_data.mime_type or "image/png"
                    elif hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                        # Fallback: if raw bytes stored, upload to GCS to externalize
                        temp_name = f"ref_{int(time.time())}.png"
                        input_image_uri = await _upload_to_gcs_async(temp_name, part.inline_data.data, part.inline_data.mime_type or "image/png")
                        input_mime_type = part.inline_data.mime_type or "image/png"
                else:
                    # Default canonical GCS path for named artifact
                    input_image_uri = f"gs://{GCS_BUCKET_NAME}/artifacts/{target_name}"

            # 3. Check recent user attachments in session events if still no URI
            if not input_image_uri and hasattr(tool_context, "session") and tool_context.session:
                events = getattr(tool_context.session, "events", [])
                for evt in reversed(events):
                    content_obj = getattr(evt, "content", None) or getattr(evt, "message", None)
                    if content_obj and getattr(content_obj, "parts", None):
                        for p in content_obj.parts:
                            if hasattr(p, "file_data") and p.file_data and p.file_data.file_uri:
                                input_image_uri = p.file_data.file_uri
                                input_mime_type = p.file_data.mime_type or "image/png"
                                break
                            elif hasattr(p, "inline_data") and p.inline_data and p.inline_data.data:
                                temp_upload_name = f"user_upload_{int(time.time())}.png"
                                input_image_uri = await _upload_to_gcs_async(temp_upload_name, p.inline_data.data, p.inline_data.mime_type or "image/png")
                                input_mime_type = p.inline_data.mime_type or "image/png"
                                break
                    if input_image_uri:
                        break

        if not input_image_uri and not input_image_bytes:
            log_user_activity(
                action="edit_image_missing_source",
                user_id=user_id,
                details={"prompt": prompt, "image_artifact_name": image_artifact_name},
            )
            return (
                "Error: No source image found to modify. Please upload an image in the chat or "
                "specify a valid `image_artifact_name` from previously generated images."
            )

        raw_bytes = await _execute_gemini_3_pro_image(
            prompt=prompt,
            input_image_uri=input_image_uri,
            input_image_bytes=input_image_bytes,
            input_mime_type=input_mime_type,
            aspect_ratio=aspect_ratio,
            image_size=image_size or "4K",
        )

        # Inject DPI metadata asynchronously in worker thread
        processed_bytes, mime_type, (width, height), (dpi_x, dpi_y) = await process_and_apply_dpi(raw_bytes)

        timestamp = int(time.time())
        filename = f"edited_{timestamp}.png"

        # Stream directly to GCS external storage
        gcs_uri = await _upload_to_gcs_async(filename, processed_bytes, mime_type)

        # Save artifact as zero-memory file_data reference in context
        if tool_context:
            artifact_file_ref = types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type)
            metadata = {
                "model": MODEL_NAME,
                "source_artifact": image_artifact_name or input_image_uri,
                "resolution": f"{width}x{height}",
                "dpi": f"{dpi_x}x{dpi_y}",
                "aspect_ratio": aspect_ratio or "auto",
                "image_size": image_size or "4K",
                "prompt": prompt,
                "user_id": user_id,
                "gcs_uri": gcs_uri,
            }
            await tool_context.save_artifact(filename=filename, artifact=artifact_file_ref, custom_metadata=metadata)

        # Explicit garbage collection
        del raw_bytes, processed_bytes
        gc.collect()

        log_user_activity(
            action="edit_image_success",
            user_id=user_id,
            details={
                "prompt": prompt,
                "source_artifact": image_artifact_name or input_image_uri,
                "artifact_filename": filename,
                "gcs_uri": gcs_uri,
                "resolution": f"{width}x{height}",
                "dpi": f"{dpi_x}x{dpi_y}",
                "aspect_ratio": aspect_ratio or "auto",
                "image_size": image_size or "4K",
            },
        )

        aspect_label = aspect_ratio if aspect_ratio else "Preserved / Auto-determined"
        return (
            f"Successfully edited image with {MODEL_NAME}:\n"
            f"- **New Artifact**: `{filename}`\n"
            f"- **GCS Storage**: `{gcs_uri}`\n"
            f"- **Source Image**: `{image_artifact_name or input_image_uri or 'Uploaded Image'}`\n"
            f"- **Model**: `{MODEL_NAME}`\n"
            f"- **Resolution**: {width}x{height} ({image_size or '4K'})\n"
            f"- **DPI Metadata**: {dpi_x} DPI\n"
            f"- **Aspect Ratio**: {aspect_label}\n"
            f"- **Edit Instructions**: \"{prompt}\""
        )
    except Exception as e:
        logger.exception("Error editing image with %s for user %s", MODEL_NAME, user_id)
        log_user_activity(
            action="edit_image_failure",
            user_id=user_id,
            details={"prompt": prompt, "error": str(e)},
        )
        return f"Failed to edit image with {MODEL_NAME}: {str(e)}"
