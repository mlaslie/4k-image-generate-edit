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
async def test_generate_4k_image_tool_file_data(mock_upload, mock_exec):
    img = Image.new("RGB", (5504, 3072), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    mock_exec.return_value = buf.getvalue()
    mock_upload.return_value = "gs://image-editing-agent-artifacts/artifacts/generated_123.png"

    mock_tool_ctx = MagicMock()
    mock_tool_ctx.user_id = "artist@agency.com"
    mock_tool_ctx.save_artifact = AsyncMock()

    result = await generate_4k_image(
        prompt="A futuristic cyber city at night",
        tool_context=mock_tool_ctx
    )

    assert "Successfully generated image" in result
    assert "5504x3072" in result
    assert "300 DPI" in result
    assert "gs://image-editing-agent-artifacts/artifacts/generated_123.png" in result
    assert mock_tool_ctx.save_artifact.called
    saved_args = mock_tool_ctx.save_artifact.call_args
    assert saved_args.kwargs["filename"].startswith("generated_")
    assert saved_args.kwargs["custom_metadata"]["gcs_uri"] == "gs://image-editing-agent-artifacts/artifacts/generated_123.png"
    # Ensure saved artifact is a file_data reference (zero raw bytes)
    saved_artifact = saved_args.kwargs["artifact"]
    assert hasattr(saved_artifact, "file_data") and saved_artifact.file_data is not None
    assert saved_artifact.file_data.file_uri == "gs://image-editing-agent-artifacts/artifacts/generated_123.png"

@pytest.mark.anyio
@patch("image_agent.tools._execute_gemini_3_pro_image")
@patch("image_agent.tools._upload_to_gcs_async")
async def test_edit_4k_image_tool_file_data(mock_upload, mock_exec):
    mock_upload.return_value = "gs://image-editing-agent-artifacts/artifacts/edited_456.png"
    part_ref = types.Part.from_uri(file_uri="gs://image-editing-agent-artifacts/artifacts/generated_123.png", mime_type="image/png")
    
    mock_tool_ctx = MagicMock()
    mock_tool_ctx.user_id = "designer@agency.com"
    mock_tool_ctx.list_artifacts = AsyncMock(return_value=["generated_123.png"])
    mock_tool_ctx.load_artifact = AsyncMock(return_value=part_ref)
    mock_tool_ctx.save_artifact = AsyncMock()

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
    assert "gs://image-editing-agent-artifacts/artifacts/edited_456.png" in result
    assert mock_tool_ctx.save_artifact.called
    saved_args = mock_tool_ctx.save_artifact.call_args
    assert saved_args.kwargs["custom_metadata"]["gcs_uri"] == "gs://image-editing-agent-artifacts/artifacts/edited_456.png"
    # Ensure saved artifact is a file_data reference
    saved_artifact = saved_args.kwargs["artifact"]
    assert hasattr(saved_artifact, "file_data") and saved_artifact.file_data is not None

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
