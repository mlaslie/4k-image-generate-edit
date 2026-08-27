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
async def test_generate_4k_image_tool_auto_aspect(mock_exec):
    img = Image.new("RGB", (5504, 3072), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    mock_exec.return_value = buf.getvalue()

    mock_tool_ctx = MagicMock()
    mock_tool_ctx.user_id = "artist@agency.com"
    mock_tool_ctx.save_artifact = AsyncMock()

    # Call without specifying aspect_ratio -> should auto-determine and default to 4K
    result = await generate_4k_image(
        prompt="A futuristic cyber city at night",
        tool_context=mock_tool_ctx
    )

    assert "Successfully generated image" in result
    assert "5504x3072" in result
    assert "300 DPI" in result
    assert "Auto-determined" in result
    assert mock_tool_ctx.save_artifact.called
    saved_args = mock_tool_ctx.save_artifact.call_args
    assert saved_args.kwargs["filename"].startswith("generated_")
    assert saved_args.kwargs["custom_metadata"]["dpi"] == "300x300"
    assert saved_args.kwargs["custom_metadata"]["model"] == MODEL_NAME
    assert saved_args.kwargs["custom_metadata"]["image_size"] == "4K"
    assert saved_args.kwargs["custom_metadata"]["aspect_ratio"] == "auto"

@pytest.mark.anyio
@patch("image_agent.tools._execute_gemini_3_pro_image")
async def test_edit_4k_image_tool(mock_exec):
    orig_img = Image.new("RGB", (4096, 4096), color="red")
    orig_buf = io.BytesIO()
    orig_img.save(orig_buf, format="PNG")
    
    part = types.Part.from_bytes(data=orig_buf.getvalue(), mime_type="image/png")
    
    mock_tool_ctx = MagicMock()
    mock_tool_ctx.user_id = "designer@agency.com"
    mock_tool_ctx.list_artifacts = AsyncMock(return_value=["original_image.png"])
    mock_tool_ctx.load_artifact = AsyncMock(return_value=part)
    mock_tool_ctx.save_artifact = AsyncMock()

    edited_img = Image.new("RGB", (4096, 4096), color="yellow")
    edited_buf = io.BytesIO()
    edited_img.save(edited_buf, format="PNG")
    mock_exec.return_value = edited_buf.getvalue()

    result = await edit_4k_image(
        prompt="Change the sky to sunset golden hour",
        image_artifact_name="original_image.png",
        aspect_ratio="1:1",
        tool_context=mock_tool_ctx
    )

    assert "Successfully edited image" in result
    assert "4096x4096" in result
    assert "300 DPI" in result
    assert mock_tool_ctx.save_artifact.called
    saved_args = mock_tool_ctx.save_artifact.call_args
    assert saved_args.kwargs["custom_metadata"]["model"] == MODEL_NAME
    assert saved_args.kwargs["custom_metadata"]["image_size"] == "4K"

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
