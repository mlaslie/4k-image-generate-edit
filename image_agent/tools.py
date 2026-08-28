import io
import os
import re
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
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "laslie-demo-project")

# Maximum number of characters of a prompt retained in an audit log entry.
MAX_LOGGED_PROMPT_CHARS = 8000

# GCS path segments are built from caller-supplied identity strings; restrict them
# to a character set that cannot escape its prefix.
_UNSAFE_SEGMENT_CHARS = re.compile(r"[^A-Za-z0-9._@+-]")


def _safe_segment(value: Optional[str], fallback: str = "unknown") -> str:
    """Sanitizes an identity string for safe use as a single GCS path segment."""
    segment = _UNSAFE_SEGMENT_CHARS.sub("_", (value or "").strip()).strip("._")
    return segment[:128] or fallback


def _user_prefix(user_id: Optional[str]) -> str:
    """Returns the per-user artifact prefix that isolates one caller's images."""
    return f"artifacts/{_safe_segment(user_id)}/"

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

    use_enterprise = (
        os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "1") == "1"
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "1") == "1"
    )
    if not use_enterprise:
        _logging_init_attempted = True
        return None, None

    try:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "laslie-demo-project")
        import google.cloud.logging

        _gcp_logging_client = google.cloud.logging.Client(project=project)
        _gcp_audit_logger = _gcp_logging_client.logger("image_agent_user_audit")
        # Only latch once a usable client exists, so a transient ADC/import failure
        # does not disable audit logging for the life of the instance.
        _logging_init_attempted = True
    except Exception as e:
        logger.warning("Cloud Logging initialization failed, will retry: %s", e)
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
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT", "laslie-demo-project")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        # Always authenticate as the Agent Runtime service account via ADC. A stray
        # GEMINI_API_KEY/GOOGLE_API_KEY in the environment must not silently switch
        # the client off Vertex AI, which would also drop image_config support.
        _genai_client = genai.Client(vertexai=True, project=project, location=location)
    return _genai_client


def get_user_identity(tool_context: Optional[ToolContext] = None) -> str:
    """
    Extracts the identity of the Gemini Enterprise end user driving this request.

    Under Agent Runtime the signed-in principal arrives as the ADK session's
    user_id, exposed as `tool_context.user_id`; the remaining lookups are
    defensive fallbacks. Returns 'unknown' rather than failing the request.
    """
    if not tool_context:
        return "unknown"

    try:
        # 1. ToolContext.user_id -> InvocationContext.user_id -> session.user_id
        user_id = getattr(tool_context, "user_id", None)
        if user_id and isinstance(user_id, str) and user_id.strip():
            return user_id.strip()

        # 2. The session object directly, if the context did not surface it
        session = getattr(tool_context, "session", None)
        if session is not None:
            session_user_id = getattr(session, "user_id", None)
            if session_user_id and isinstance(session_user_id, str) and session_user_id.strip():
                return session_user_id.strip()

        # 3. Session state, which is a State/Mapping rather than a plain dict
        state = getattr(tool_context, "state", None)
        if state is not None:
            for key in ["user_email", "email", "user_id", "user", "preferred_username"]:
                try:
                    val = state.get(key)
                except Exception:
                    val = None
                if val and isinstance(val, str) and val.strip():
                    return val.strip()

        # 4. Invocation custom_metadata
        custom_metadata = getattr(tool_context, "custom_metadata", None)
        if custom_metadata:
            for key in ["user_email", "email", "user_id", "user", "principal"]:
                val = custom_metadata.get(key)
                if val and isinstance(val, str) and val.strip():
                    return val.strip()
    except Exception:
        pass

    return "unknown"


def _get_session_id(tool_context: Optional[ToolContext] = None) -> str:
    """Extracts session_id strictly from tool_context or session."""
    if not tool_context:
        return ""
    if hasattr(tool_context, "session") and tool_context.session:
        sid = (
            getattr(tool_context.session, "id", None)
            or getattr(tool_context.session, "session_id", None)
            or getattr(tool_context.session, "name", None)
        )
        if sid:
            return str(sid).split("/")[-1]
    if hasattr(tool_context, "session_id") and tool_context.session_id:
        return str(tool_context.session_id).split("/")[-1]
    return ""


def _context_ids(tool_context: Optional[ToolContext] = None) -> Dict[str, str]:
    """
    Returns the session and invocation identifiers for the current request, so an
    audit entry can be correlated with the runtime logs and the stored session.
    """
    ids = {"session_id": _get_session_id(tool_context)}
    if tool_context is not None:
        invocation_id = getattr(tool_context, "invocation_id", None)
        if invocation_id:
            ids["invocation_id"] = str(invocation_id)
    return ids


def _write_cloud_log_struct(log_payload: Dict[str, Any]):
    """Helper to write log entry with ReasoningEngine resource attachment in worker thread."""
    try:
        _, gcp_logger = get_logging_client()
        if gcp_logger:
            from google.cloud.logging_v2.resource import Resource

            # Agent Runtime injects GOOGLE_CLOUD_AGENT_ENGINE_ID/_LOCATION. Never fall
            # back to a hardcoded engine id: that would file this agent's audit trail
            # under a different engine. GOOGLE_CLOUD_LOCATION is the model API region
            # ("global") and is deliberately not used for the engine resource.
            engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID") or os.environ.get(
                "REASONING_ENGINE_ID", ""
            )
            location = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION", "")
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "laslie-demo-project")

            resource = None
            if engine_id and location:
                resource = Resource(
                    type="aiplatform.googleapis.com/ReasoningEngine",
                    labels={
                        "reasoning_engine_id": engine_id,
                        "location": location,
                        # Bare project identifier, matching the runtime's own entries.
                        "resource_container": project,
                    },
                )
            if resource is not None:
                gcp_logger.log_struct(log_payload, resource=resource, severity="INFO")
            else:
                gcp_logger.log_struct(log_payload, severity="INFO")
    except Exception as e:
        logger.debug("Cloud Logging struct write exception: %s", e)


def log_user_activity(action: str, user_id: str, details: Dict[str, Any]):
    """
    Logs user activity to standard logging and Google Cloud Logging via singleton and ReasoningEngine resource.
    Guaranteed never to raise exceptions or interrupt tool execution.
    """
    try:
        safe_user_id = user_id if (user_id and isinstance(user_id, str) and user_id.strip()) else "unknown"
        safe_details = dict(details)
        for key in ("prompt", "negative_prompt", "error"):
            val = safe_details.get(key)
            if isinstance(val, str) and len(val) > MAX_LOGGED_PROMPT_CHARS:
                safe_details[key] = val[:MAX_LOGGED_PROMPT_CHARS] + "...[truncated]"
        log_payload = {
            "event": "image_agent_user_activity",
            "action": action,
            "user_id": safe_user_id,
            "timestamp": time.time(),
            **safe_details,
        }

        logger.info(
            "User [%s] performed action '%s' - Details: %s",
            safe_user_id,
            action,
            json.dumps(log_payload, default=str),
        )

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


def _describe_empty_response(response: Any) -> str:
    """Summarizes why a response carried no image, for logs and user-facing errors."""
    details = []
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)
    if block_reason:
        details.append(f"block_reason={block_reason}")
    for cand in getattr(response, "candidates", None) or []:
        finish_reason = getattr(cand, "finish_reason", None)
        if finish_reason:
            details.append(f"finish_reason={finish_reason}")
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                details.append(f"text={text[:200]}")
    return "; ".join(details) or "no reason reported"


def _extract_image_bytes(response: Any) -> Optional[bytes]:
    """Helper to extract raw image bytes from a GenerateContent response."""
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if part.inline_data and part.inline_data.data:
                        return part.inline_data.data
    return None


async def _upload_to_gcs_async(
    object_path: str, data: bytes, mime_type: str
) -> str:
    """
    Uploads raw image bytes to `object_path` in the artifact bucket and returns the
    gs:// URI. Raises on failure: callers must not report success, or hand back a
    download link, for an object that was never written.
    """

    def _sync_upload():
        client = get_storage_client()
        if client is None:
            raise RuntimeError("Google Cloud Storage client is unavailable")
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(object_path)
        blob.upload_from_string(data, content_type=mime_type)
        return f"gs://{GCS_BUCKET_NAME}/{object_path}"

    return await asyncio.to_thread(_sync_upload)


async def _resolve_source_image_uri(
    image_artifact_name: str,
    user_id: str,
    tool_context: Optional[ToolContext] = None,
) -> Tuple[Optional[str], Optional[bytes], str]:
    """
    Resolves the source image to edit, restricted to images belonging to the calling
    user. Resolution order:

    1. An explicitly named artifact under this user's prefix, or a gs:// URI that
       already lies under that prefix.
    2. An image the user attached on the current turn (paste or upload).
    3. The most recent image in a user-authored event of the current session.
    4. An ADK-managed artifact loaded through the tool context.

    Returns (None, None, "image/png") when the user has no image in scope, so the
    caller can ask them to supply one rather than guessing at a URI.
    """
    prefix = _user_prefix(user_id)
    safe_user = _safe_segment(user_id)
    safe_session = _safe_segment(_get_session_id(tool_context), fallback="nosession")
    client = get_storage_client()
    bucket = client.bucket(GCS_BUCKET_NAME) if client else None

    # Priority 1: an explicitly named source. An absolute gs:// URI is honoured only
    # when it addresses this user's own prefix, so a crafted prompt cannot reach
    # another user's images.
    if image_artifact_name:
        if image_artifact_name.startswith("gs://"):
            allowed_root = f"gs://{GCS_BUCKET_NAME}/{prefix}"
            if image_artifact_name.startswith(allowed_root):
                return image_artifact_name, None, "image/png"
            logger.warning("Rejected out-of-scope source image URI for user %s", user_id)
            return None, None, "image/png"

        if bucket:
            # Exact object names only; a substring match would let one artifact name
            # select a different user's or a different session's image.
            candidates = [f"{prefix}{image_artifact_name}"]
            if not image_artifact_name.endswith(".png"):
                candidates.append(f"{prefix}{image_artifact_name}.png")
            for candidate in candidates:
                if bucket.blob(candidate).exists():
                    return f"gs://{GCS_BUCKET_NAME}/{candidate}", None, "image/png"

    # Priority 2: an image attached by the user on this turn.
    if tool_context is not None:
        user_content = getattr(tool_context, "user_content", None)
        resolved = await _image_from_content(user_content, safe_user, safe_session)
        if resolved:
            return resolved

    # Priority 3: the most recent image the user contributed earlier in this session.
    if tool_context is not None:
        session = getattr(tool_context, "session", None)
        for evt in reversed(getattr(session, "events", []) or []):
            # Only user-authored events: an agent event could carry an image the
            # user never chose as the edit source.
            if getattr(evt, "author", None) != "user":
                continue
            content_obj = getattr(evt, "content", None) or getattr(evt, "message", None)
            resolved = await _image_from_content(content_obj, safe_user, safe_session)
            if resolved:
                return resolved

    # Priority 4: an ADK-managed artifact in the current session.
    if image_artifact_name and tool_context is not None and hasattr(tool_context, "load_artifact"):
        try:
            part = await tool_context.load_artifact(image_artifact_name)
            resolved = await _image_from_content(
                types.Content(parts=[part]) if part else None, safe_user, safe_session
            )
            if resolved:
                return resolved
        except Exception as e:
            logger.debug("load_artifact lookup failed for %s: %s", image_artifact_name, e)

    # Nothing in scope for this user: never invent a URI.
    return None, None, "image/png"


async def _image_from_content(
    content_obj: Any, safe_user: str, safe_session: str
) -> Optional[Tuple[str, Optional[bytes], str]]:
    """
    Returns a (uri, bytes, mime_type) triple for the first image part of `content_obj`,
    staging inline bytes into this user's own upload prefix. Returns None if there is
    no image part.
    """
    parts = getattr(content_obj, "parts", None) if content_obj else None
    if not parts:
        return None

    for p in parts:
        file_data = getattr(p, "file_data", None)
        if file_data and getattr(file_data, "file_uri", None):
            return file_data.file_uri, None, file_data.mime_type or "image/png"

        inline_data = getattr(p, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            mime_type = inline_data.mime_type or "image/png"
            object_path = (
                f"uploads/{safe_user}/{safe_session}/pasted_{int(time.time() * 1000)}.png"
            )
            uri = await _upload_to_gcs_async(object_path, inline_data.data, mime_type)
            return uri, None, mime_type

    return None


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
        contents.append(types.Part.from_uri(file_uri=input_image_uri, mime_type=input_mime_type))
    elif input_image_bytes:
        contents.append(types.Part.from_bytes(data=input_image_bytes, mime_type=input_mime_type))

    contents.append(prompt)

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
            # The call succeeded but carried no image (typically a safety block).
            # Retrying will not change that, so surface the model's own reason.
            raise RuntimeError(
                f"{MODEL_NAME} returned no image ({_describe_empty_response(response)})"
            )
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
    Saves image directly to GCS without registering artifact_delta.

    Args:
        prompt: Detailed description of the image to generate.
        aspect_ratio: Aspect ratio of the output image (e.g., '1:1', '16:9', '9:16', '4:3', '3:4', '21:9', '3:2', '2:3').
                      If omitted or empty, the model automatically determines the best composition aspect ratio.
        image_size: Target resolution ('4K', '2K', '1K'). Defaults to '4K'.
        negative_prompt: Optional elements to exclude from the generated image.
        tool_context: ADK ToolContext for user identity and session management.

    Returns:
        A message detailing the generated image, resolution, DPI metadata, clickable download links, and artifact filename.
    """
    user_id = get_user_identity(tool_context)
    context_ids = _context_ids(tool_context)
    started = time.monotonic()
    log_user_activity(
        action="generate_image_request",
        user_id=user_id,
        details={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio or "auto",
            "image_size": image_size or "4K",
            "negative_prompt": negative_prompt,
            **context_ids,
        },
    )

    try:
        full_prompt = f"{prompt} (Exclude: {negative_prompt})" if negative_prompt else prompt

        # Stage timings are recorded so slow turns can be attributed to the model
        # call, the DPI re-encode, or the upload without extra instrumentation.
        stage_start = time.monotonic()
        raw_bytes = await _execute_gemini_3_pro_image(
            prompt=full_prompt,
            aspect_ratio=aspect_ratio,
            image_size=image_size or "4K",
        )
        model_ms = int((time.monotonic() - stage_start) * 1000)

        # Inject DPI metadata asynchronously in worker thread
        stage_start = time.monotonic()
        processed_bytes, mime_type, (width, height), (dpi_x, dpi_y) = await process_and_apply_dpi(raw_bytes)
        dpi_ms = int((time.monotonic() - stage_start) * 1000)

        timestamp = int(time.time())
        filename = f"generated_{timestamp}.png"
        object_path = f"{_user_prefix(user_id)}{filename}"

        # Stream directly to GCS external storage. A failed upload raises, so a
        # download link is never reported for an object that does not exist.
        stage_start = time.monotonic()
        gcs_uri = await _upload_to_gcs_async(object_path, processed_bytes, mime_type)
        upload_ms = int((time.monotonic() - stage_start) * 1000)

        # Direct clickable HTTPS download URL and Console explorer URL
        https_view_url = f"https://storage.cloud.google.com/{GCS_BUCKET_NAME}/{object_path}"
        console_url = f"https://console.cloud.google.com/storage/browser/_details/{GCS_BUCKET_NAME}/{object_path}?project={GCP_PROJECT}"

        log_user_activity(
            action="generate_image_success",
            user_id=user_id,
            details={
                "prompt": prompt,
                "artifact_filename": filename,
                "gcs_uri": gcs_uri,
                "https_url": https_view_url,
                "resolution": f"{width}x{height}",
                "dpi": f"{dpi_x}x{dpi_y}",
                "aspect_ratio": aspect_ratio or "auto",
                "image_size": image_size or "4K",
                "duration_ms": int((time.monotonic() - started) * 1000),
                "model_ms": model_ms,
                "dpi_ms": dpi_ms,
                "upload_ms": upload_ms,
                "image_bytes": len(processed_bytes),
                **context_ids,
            },
        )

        aspect_label = aspect_ratio if aspect_ratio else "Auto-determined by model"
        return (
            f"Successfully generated image with {MODEL_NAME}:\n"
            f"- **Direct Download / View**: [{filename}]({https_view_url})\n"
            f"- **GCS Console**: [Open in Google Cloud Console]({console_url})\n"
            f"- **GCS URI**: `{gcs_uri}`\n"
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
            details={
                "prompt": prompt,
                "error": str(e),
                "duration_ms": int((time.monotonic() - started) * 1000),
                **context_ids,
            },
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
    Locates user uploaded photos, pasted images, or previously generated images strictly within the current session.

    Args:
        prompt: Instructions for modifying or transforming the image.
        image_artifact_name: The filename/artifact name or gs:// URI of the base image to modify.
                            If omitted, the uploaded/pasted image from the current session will be used.
        aspect_ratio: Desired output aspect ratio (e.g., '1:1', '16:9', '9:16', '4:3', '3:4', '21:9', '3:2', '2:3').
                      If omitted or empty, preserves / auto-determines aspect ratio.
        image_size: Target resolution ('4K', '2K', '1K'). Defaults to '4K'.
        tool_context: ADK ToolContext for session events and user identity.

    Returns:
        A message detailing the edited image, resolution, DPI metadata, clickable download links, and new artifact filename.
    """
    user_id = get_user_identity(tool_context)
    context_ids = _context_ids(tool_context)
    started = time.monotonic()
    log_user_activity(
        action="edit_image_request",
        user_id=user_id,
        details={
            "prompt": prompt,
            "image_artifact_name": image_artifact_name,
            "aspect_ratio": aspect_ratio or "auto",
            "image_size": image_size or "4K",
            **context_ids,
        },
    )

    try:
        # Stage timings are recorded so slow turns can be attributed to source
        # resolution, the model call, the DPI re-encode, or the upload.
        stage_start = time.monotonic()
        input_image_uri, input_image_bytes, input_mime_type = await _resolve_source_image_uri(
            image_artifact_name=image_artifact_name,
            user_id=user_id,
            tool_context=tool_context,
        )
        resolve_ms = int((time.monotonic() - stage_start) * 1000)

        if not input_image_uri and not input_image_bytes:
            log_user_activity(
                action="edit_image_missing_source",
                user_id=user_id,
                details={
                    "prompt": prompt,
                    "image_artifact_name": image_artifact_name,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    **context_ids,
                },
            )
            return (
                "Error: No image was found in this conversation to edit. "
                "Please upload or paste an image into the chat, or provide a prompt to generate a new 4K image."
            )

        stage_start = time.monotonic()
        raw_bytes = await _execute_gemini_3_pro_image(
            prompt=prompt,
            input_image_uri=input_image_uri,
            input_image_bytes=input_image_bytes,
            input_mime_type=input_mime_type,
            aspect_ratio=aspect_ratio,
            image_size=image_size or "4K",
        )
        model_ms = int((time.monotonic() - stage_start) * 1000)

        # Inject DPI metadata asynchronously in worker thread
        stage_start = time.monotonic()
        processed_bytes, mime_type, (width, height), (dpi_x, dpi_y) = await process_and_apply_dpi(raw_bytes)
        dpi_ms = int((time.monotonic() - stage_start) * 1000)

        timestamp = int(time.time())
        filename = f"edited_{timestamp}.png"
        object_path = f"{_user_prefix(user_id)}{filename}"

        # Stream directly to GCS external storage. A failed upload raises, so a
        # download link is never reported for an object that does not exist.
        stage_start = time.monotonic()
        gcs_uri = await _upload_to_gcs_async(object_path, processed_bytes, mime_type)
        upload_ms = int((time.monotonic() - stage_start) * 1000)

        # Direct clickable HTTPS download URL and Console explorer URL
        https_view_url = f"https://storage.cloud.google.com/{GCS_BUCKET_NAME}/{object_path}"
        console_url = f"https://console.cloud.google.com/storage/browser/_details/{GCS_BUCKET_NAME}/{object_path}?project={GCP_PROJECT}"

        log_user_activity(
            action="edit_image_success",
            user_id=user_id,
            details={
                "prompt": prompt,
                "source_artifact": image_artifact_name or input_image_uri,
                "artifact_filename": filename,
                "gcs_uri": gcs_uri,
                "https_url": https_view_url,
                "resolution": f"{width}x{height}",
                "dpi": f"{dpi_x}x{dpi_y}",
                "aspect_ratio": aspect_ratio or "auto",
                "image_size": image_size or "4K",
                "duration_ms": int((time.monotonic() - started) * 1000),
                "resolve_ms": resolve_ms,
                "model_ms": model_ms,
                "dpi_ms": dpi_ms,
                "upload_ms": upload_ms,
                "image_bytes": len(processed_bytes),
                **context_ids,
            },
        )

        aspect_label = aspect_ratio if aspect_ratio else "Preserved / Auto-determined"
        return (
            f"Successfully edited image with {MODEL_NAME}:\n"
            f"- **Direct Download / View**: [{filename}]({https_view_url})\n"
            f"- **GCS Console**: [Open in Google Cloud Console]({console_url})\n"
            f"- **GCS URI**: `{gcs_uri}`\n"
            f"- **Source Image**: `{image_artifact_name or input_image_uri or 'Uploaded / Pasted Image'}`\n"
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
            details={
                "prompt": prompt,
                "error": str(e),
                "duration_ms": int((time.monotonic() - started) * 1000),
                **context_ids,
            },
        )
        return f"Failed to edit image with {MODEL_NAME}: {str(e)}"
