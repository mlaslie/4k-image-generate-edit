import io
import pytest
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types
from image_agent.tools import (
    calculate_dpi_for_resolution,
    process_and_apply_dpi,
    generate_4k_image,
    edit_4k_image,
    get_user_identity,
    MODEL_NAME,
)
from image_agent.agent import root_agent

def test_calculate_dpi_for_resolution():
    assert calculate_dpi_for_resolution(3840, 2160) == (300, 300)
    assert calculate_dpi_for_resolution(4096, 4096) == (300, 300)
    assert calculate_dpi_for_resolution(5504, 3072) == (300, 300)
    assert calculate_dpi_for_resolution(2048, 1536) == (240, 240)
    assert calculate_dpi_for_resolution(1024, 1024) == (150, 150)
    assert calculate_dpi_for_resolution(800, 600) == (72, 72)

@pytest.mark.anyio
async def test_process_and_apply_dpi():
    img = Image.new("RGB", (3840, 2160), color="purple")
    raw_buf = io.BytesIO()
    img.save(raw_buf, format="PNG")
    raw_bytes = raw_buf.getvalue()

    processed_bytes, mime_type, (w, h), (dpi_x, dpi_y) = await process_and_apply_dpi(raw_bytes)
    assert w == 3840
    assert h == 2160
    assert dpi_x == 300
    assert dpi_y == 300
    assert mime_type == "image/png"

    reloaded = Image.open(io.BytesIO(processed_bytes))
    dpi_info = reloaded.info.get("dpi")
    assert dpi_info is not None
    assert round(dpi_info[0]) == 300
    assert round(dpi_info[1]) == 300

def test_root_agent_definition():
    assert root_agent.name == "image_agent"
    assert len(root_agent.tools) == 2
    tool_names = [t.__name__ for t in root_agent.tools]
    assert "generate_4k_image" in tool_names
    assert "edit_4k_image" in tool_names

@pytest.mark.anyio
@patch("image_agent.tools._execute_gemini_3_pro_image")
@patch("image_agent.tools._upload_to_gcs_async")
async def test_generate_4k_image_tool_no_artifact_delta(mock_upload, mock_exec):
    img = Image.new("RGB", (5504, 3072), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    mock_exec.return_value = buf.getvalue()
    mock_upload.return_value = "gs://image-editing-agent-artifacts/artifacts/artist@agency.com/generated_123.png"

    mock_tool_ctx = MagicMock()
    mock_tool_ctx.user_id = "artist@agency.com"

    result = await generate_4k_image(
        prompt="A futuristic cyber city at night",
        tool_context=mock_tool_ctx
    )

    assert "Successfully generated image" in result
    assert "5504x3072" in result
    assert "300 DPI" in result
    assert "gs://image-editing-agent-artifacts/artifacts/artist@agency.com/generated_123.png" in result
    assert (
        "https://storage.cloud.google.com/image-editing-agent-artifacts/"
        "artifacts/artist@agency.com/generated_" in result
    )
    # The object is written under the caller's own prefix, not a shared namespace.
    assert mock_upload.call_args[0][0].startswith("artifacts/artist@agency.com/")

@pytest.mark.anyio
@patch("image_agent.tools._execute_gemini_3_pro_image")
@patch("image_agent.tools._upload_to_gcs_async")
async def test_edit_4k_image_tool_no_artifact_delta(mock_upload, mock_exec):
    mock_upload.return_value = "gs://image-editing-agent-artifacts/artifacts/designer@agency.com/edited_456.png"

    source_img = Image.new("RGB", (1024, 1024), color="green")
    source_buf = io.BytesIO()
    source_img.save(source_buf, format="PNG")

    mock_tool_ctx = MagicMock()
    mock_tool_ctx.user_id = "designer@agency.com"
    mock_tool_ctx.user_content = types.Content(
        parts=[types.Part.from_bytes(data=source_buf.getvalue(), mime_type="image/png")]
    )

    edited_img = Image.new("RGB", (4096, 4096), color="yellow")
    edited_buf = io.BytesIO()
    edited_img.save(edited_buf, format="PNG")
    mock_exec.return_value = edited_buf.getvalue()

    result = await edit_4k_image(
        prompt="Change the sky to sunset golden hour",
        image_artifact_name="generated_123.png",
        aspect_ratio="1:1",
        tool_context=mock_tool_ctx
    )

    assert "Successfully edited image" in result
    assert "4096x4096" in result
    assert "300 DPI" in result
    assert "gs://image-editing-agent-artifacts/artifacts/designer@agency.com/edited_456.png" in result
    assert (
        "https://storage.cloud.google.com/image-editing-agent-artifacts/"
        "artifacts/designer@agency.com/edited_" in result
    )

def test_get_user_identity_variations():
    assert get_user_identity(None) == "unknown"

    mock_ctx = MagicMock()
    mock_ctx.user_id = "test.user@company.com"
    assert get_user_identity(mock_ctx) == "test.user@company.com"

    mock_ctx2 = MagicMock()
    mock_ctx2.user_id = None
    mock_ctx2.session.user_id = "session.user@company.com"
    assert get_user_identity(mock_ctx2) == "session.user@company.com"

    mock_ctx3 = MagicMock()
    mock_ctx3.user_id = "  "
    mock_ctx3.session = None
    mock_ctx3.custom_metadata = None
    assert get_user_identity(mock_ctx3) == "unknown"


@pytest.mark.anyio
async def test_edit_without_any_image_asks_the_user():
    """A named artifact that does not exist must not resolve to an invented URI."""
    mock_tool_ctx = MagicMock()
    mock_tool_ctx.user_id = "designer@agency.com"
    mock_tool_ctx.user_content = None
    mock_tool_ctx.session.events = []
    mock_tool_ctx.load_artifact = AsyncMock(return_value=None)

    with patch("image_agent.tools.get_storage_client", return_value=None):
        result = await edit_4k_image(
            prompt="make it snow",
            image_artifact_name="generated_does_not_exist.png",
            tool_context=mock_tool_ctx,
        )

    assert "No image was found" in result


@pytest.mark.anyio
async def test_gs_uri_outside_callers_prefix_is_rejected():
    """A prompt-supplied gs:// URI for another user's image must not resolve."""
    from image_agent.tools import _resolve_source_image_uri

    mock_tool_ctx = MagicMock()
    mock_tool_ctx.user_content = None
    mock_tool_ctx.session.events = []
    mock_tool_ctx.load_artifact = AsyncMock(return_value=None)

    uri, data, _ = await _resolve_source_image_uri(
        image_artifact_name="gs://image-editing-agent-artifacts/artifacts/victim@agency.com/generated_1.png",
        user_id="attacker@agency.com",
        tool_context=mock_tool_ctx,
    )
    assert uri is None and data is None


@pytest.mark.anyio
async def test_failed_upload_does_not_report_success():
    """A GCS write failure must surface, not yield a link to a nonexistent object."""
    img = Image.new("RGB", (3840, 2160), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    with patch("image_agent.tools._execute_gemini_3_pro_image", AsyncMock(return_value=buf.getvalue())), \
         patch("image_agent.tools.get_storage_client", return_value=None):
        result = await generate_4k_image(prompt="a red square", tool_context=MagicMock())

    assert "Failed to generate image" in result
    assert "storage.cloud.google.com" not in result


def test_audit_log_resource_uses_agent_engine_env(monkeypatch):
    """Audit entries must attach to this engine, never a hardcoded or model-API region."""
    from image_agent import tools

    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "1234567890")
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION", "us-central1")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "laslie-demo-project")

    captured = {}

    class FakeLogger:
        def log_struct(self, payload, **kwargs):
            captured["payload"] = payload
            captured["resource"] = kwargs.get("resource")

    monkeypatch.setattr(tools, "get_logging_client", lambda: (object(), FakeLogger()))
    tools._write_cloud_log_struct({"event": "test"})

    labels = captured["resource"].labels
    assert labels["reasoning_engine_id"] == "1234567890"
    assert labels["location"] == "us-central1"
    assert labels["resource_container"] == "laslie-demo-project"


def test_long_prompts_are_truncated_in_audit_log(monkeypatch):
    from image_agent import tools

    captured = {}
    monkeypatch.setattr(tools, "_write_cloud_log_struct", lambda payload: captured.update(payload))
    tools.log_user_activity("generate_image_request", "u@example.com", {"prompt": "x" * 20000})

    assert len(captured["prompt"]) < 20000
    assert captured["prompt"].endswith("...[truncated]")


@pytest.mark.anyio
async def test_success_audit_entry_carries_correlation_and_timings(monkeypatch):
    """Audit entries must be correlatable to a session and attribute turn latency."""
    from image_agent import tools

    entries = []
    monkeypatch.setattr(tools, "log_user_activity",
                        lambda action, user_id, details: entries.append((action, details)))

    img = Image.new("RGB", (3840, 2160), color="teal")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    mock_ctx = MagicMock()
    mock_ctx.user_id = "timing@agency.com"
    mock_ctx.session.id = "sess-123"
    mock_ctx.invocation_id = "inv-456"

    with patch("image_agent.tools._execute_gemini_3_pro_image", AsyncMock(return_value=buf.getvalue())), \
         patch("image_agent.tools._upload_to_gcs_async",
               AsyncMock(return_value="gs://image-editing-agent-artifacts/artifacts/timing@agency.com/generated_1.png")):
        await tools.generate_4k_image(prompt="a teal square", tool_context=mock_ctx)

    actions = dict(entries)
    assert actions["generate_image_request"]["session_id"] == "sess-123"
    assert actions["generate_image_request"]["invocation_id"] == "inv-456"

    success = actions["generate_image_success"]
    assert success["session_id"] == "sess-123"
    assert success["invocation_id"] == "inv-456"
    for field in ("duration_ms", "model_ms", "dpi_ms", "upload_ms", "image_bytes"):
        assert isinstance(success[field], int), field
    # Total turn time must account for at least the individual stages.
    assert success["duration_ms"] >= success["dpi_ms"]
