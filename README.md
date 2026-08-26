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

To connect the deployed Agent Runtime agent to **Gemini Enterprise**:

1. **Authorization Resource (OAuth 2.0)**:
   - **Authorization URL**:
     ```text
     https://accounts.google.com/o/oauth2/v2/auth?response_type=code&access_type=offline&prompt=consent&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform+openid+email+profile
     ```
   - **Token URL**: `https://oauth2.googleapis.com/token`
   - **Scopes**: `https://www.googleapis.com/auth/cloud-platform openid email profile`

2. **Agent Resource Path**:
   Provide the full resource name returned from the deployment output:
   ```text
   projects/YOUR_PROJECT_ID/locations/us-central1/reasoningEngines/YOUR_REASONING_ENGINE_ID
   ```

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
