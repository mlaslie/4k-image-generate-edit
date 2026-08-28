# 4K Image Generation & Editing ADK Agent

An enterprise-ready AI agent built with the **Google Agent Development Kit (ADK)** that generates and modifies high-fidelity **4K UHD** images with print-standard **300 DPI** metadata injection, powered by **Gemini 3.7 Flash** (orchestrator) and **Gemini 3 Pro Image** on Vertex AI.

---

## 🌟 Key Features

- **High-Fidelity 4K Output**: Generates crisp, high-resolution visuals across multiple aspect ratios (`1:1` 4096x4096, `16:9` 3840x2160, `9:16` 2160x3840, `4:3` 3840x2880, `3:4` 2880x3840).
- **Multimodal Image Editing**: Modifies existing uploaded images or session artifacts using natural language instructions.
- **DPI Metadata Embedding**: Automatically calculates and injects 300 DPI physical pixel density metadata into PNG `pHYs` chunks for print-ready assets.
- **Gemini Enterprise User Audit Logging**: Asynchronously logs Gemini Enterprise user identity (`user_id`, email, or OIDC claims) and operation metadata to Google Cloud Logging (`image_agent_user_audit`) without blocking or failing requests.
- **Enterprise Ready**: Full support for Google Cloud Agent Runtime (Vertex AI Reasoning Engines / Agent Engine), OpenTelemetry distributed tracing & logging, and Google Application Default Credentials (ADC).

---

## 📁 Repository Structure

```
.
├── image_agent/
│   ├── __init__.py          # Package initialization (exports root_agent)
│   ├── agent.py             # Root agent orchestrator definition (gemini-3.7-flash)
│   ├── tools.py             # 4K image generation, editing, DPI & Cloud Logging tools
│   ├── requirements.txt     # Container build dependencies
│   └── .env                 # Environment variables for the agent package
├── test_image_agent.py      # Automated unit & integration tests (pytest)
├── requirements.txt         # Root dependency manifest
├── .env.example             # Template for local environment variables
└── README.md                # Documentation & deployment guide
```

---

## 🚀 Quick Start: Running Locally

### 1. Prerequisites
- Python 3.11+ (Python 3.11 - 3.13 supported)
- Google Cloud SDK (`gcloud`) installed and configured
- Vertex AI API enabled in your GCP project

### 2. Authenticate with Google Cloud
Ensure Application Default Credentials (ADC) are set up for your project:
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Environment
Create `.env` from the example template:
```bash
cp .env.example .env
cp .env.example image_agent/.env
```
Ensure `.env` contains:
```env
GOOGLE_GENAI_USE_ENTERPRISE=1
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
GOOGLE_CLOUD_LOCATION=global
AGENT_LLM_MODEL=gemini-3.7-flash
```

### 5. Run with ADK Web UI
Launch the interactive local ADK web interface:
```bash
adk web .
```
Open your browser at `http://127.0.0.1:8000`, select **`image_agent`**, and test generating or editing 4K images.

### 6. Run Unit Tests
```bash
pytest test_image_agent.py -v
```

---

## ☁️ Deploying to Google Cloud Agent Runtime

Deploy the agent to **Google Cloud Agent Runtime (Vertex AI Reasoning Engines)** with OpenTelemetry telemetry enabled:

### In-Place Update or Initial Deploy
```bash
adk deploy agent_engine image_agent \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --display_name "4k-image-generation-alt" \
  --description "4K Image Generation & Editing Agent with Gemini 3.7 Flash and Gemini 3 Pro Image" \
  --otel_to_cloud
```

*Note: To update an existing instance in place, add `--agent_engine_id <EXISTING_REASONING_ENGINE_ID>`.*

---

## 🔐 Configuring in Gemini Enterprise

Note: The values below are for Google OAuth. If using a different OAuth provider, the values will be different.

### Authorization Resource
- **Authorization Name**: A name of your choosing
- **Client ID**: Obtained from your OAuth Provider
- **Client Secret**: Obtained from your OAuth Provider
- **Token URI**: `https://oauth2.googleapis.com/token`
- **Authorization URI**: `https://accounts.google.com/o/oauth2/v2/auth?response_type=code&access_type=offline&prompt=consent&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform+openid+email+profile`

### Agent Definition
- **Agent Name**: A name of your choosing
- **Agent Description**: A description of your choosing
- **Agent Runtime Reasoning Engine**: `projects/YOUR_PROJECT_ID/locations/YOUR_REGION/reasoningEngines/YOUR_REASONING_ENGINE_ID`

---

## 🔄 Using This Agent Alongside Gemini Enterprise Image Generation

Gemini Enterprise can create, edit, and refine images at **1K** resolution using the
image capability included in your subscription. This agent complements that: when you
want to **upscale to 2K or 4K**, or **change the aspect ratio**, `@`-mention the agent
in the same conversation.

**One extra step is required: attach the image to your message.**

Images produced by Gemini Enterprise's built-in image generation are not shared with an
`@`-mentioned agent. The conversation the agent receives carries only a short line of
text — for example `Image generated by Nano Banana Pro.` — and never the picture itself.
The agent has no way to reach it.

So before asking for the upscale, put the image into your message: **copy and paste it
into the chat**, or **download it and upload it**. Once attached, the agent renders it at
the requested resolution or aspect ratio while preserving the original composition,
colors, lighting, and subject.

If you skip that step, the agent will tell you it cannot see the image and ask you to
provide it. It will not quietly produce a substitute: an image reconstructed from the
surrounding conversation text would resemble yours without actually being it.

> **Tip:** If you know up front that you want 4K, ask the agent to create the image
> directly. It generates at 4K in a single step, which avoids the round trip entirely.

---

## 📊 Observability & Audit Logs

When deployed, all image generation and modification requests are recorded to Google Cloud Logging under:
- **Log Name**: `projects/YOUR_PROJECT_ID/logs/image_agent_user_audit`
- **Resource**: `aiplatform.googleapis.com/ReasoningEngine`

To view the audit logs in GCP Logs Explorer:
```sql
resource.type="aiplatform.googleapis.com/ReasoningEngine"
logName="projects/YOUR_PROJECT_ID/logs/image_agent_user_audit"
```

Each entry records the requesting user, the prompt, and the link to the output image,
along with correlation and timing fields:

| Field | Description |
|---|---|
| `user_id` | The signed-in Gemini Enterprise user making the request |
| `prompt` | The generation or edit instruction (truncated at 8,000 characters) |
| `gcs_uri` / `https_url` | The output image |
| `session_id` / `invocation_id` | Correlates the entry with the stored session and the runtime logs |
| `duration_ms` | Total time spent in the tool |
| `model_ms`, `dpi_ms`, `upload_ms`, `resolve_ms` | Per-stage breakdown of that time |

To trace a single turn end to end:
```sql
logName="projects/YOUR_PROJECT_ID/logs/image_agent_user_audit"
jsonPayload.session_id="YOUR_SESSION_ID"
```

---

## 🩺 Troubleshooting

### A request in the Gemini Enterprise app appears to hang or never returns a result

**Refresh the browser.** The result is usually already there.

4K image generation and editing take roughly 50–70 seconds per turn, during which the
agent sends nothing on the streaming connection. If that silent gap exceeds the client's
tolerance, the Gemini Enterprise app stops rendering the response even though the agent
completed the work normally.

The agent always writes its final answer to the session before finishing, so refreshing
the conversation re-reads it from session state and the result appears. Nothing is lost,
and the request does not need to be resubmitted.

To confirm a turn actually completed, check for a matching success entry:
```sql
logName="projects/YOUR_PROJECT_ID/logs/image_agent_user_audit"
jsonPayload.action=("generate_image_success" OR "edit_image_success")
```
If the entry is present, the image was generated and uploaded successfully, and
`https_url` links directly to it.

### The agent says it cannot see an image that Gemini Enterprise just created

This is expected, and the message is accurate. Images created by Gemini Enterprise's
built-in image generation are not passed to an `@`-mentioned agent — only a short text
note about them appears in the conversation. Copy and paste the image into the chat, or
download and re-upload it, and the request will complete. See
[Using This Agent Alongside Gemini Enterprise Image Generation](#-using-this-agent-alongside-gemini-enterprise-image-generation).

### The first request after a period of inactivity is dropped

Set `min_instances` to `1` in `.agent_engine_config.json`. With `min_instances: 0` the
container scales to zero, and a request arriving during the ~90 second cold start can be
dropped before it ever reaches the agent.
