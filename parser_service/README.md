# Qwen NLU Parser Microservice (FastAPI)

This folder exposes your fine-tuned Qwen 2.5 + LoRA adapter as an HTTP API so your LangGraph agent can replace Gemini parsing.

## What it returns

`POST /parse`

```json
{
  "intent": "routing",
  "origin": "سموحة",
  "destination": "محطة الرمل",
  "mode": null,
  "constraints": [],
  "query_type": null
}
```

## Local run (Windows)

1) Create venv and install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r parser_service\requirements.txt
```

2) Run:

```powershell
.\parser_service\run_local.ps1 -Port 8000
```

Notes:
- Qwen 2.5 3B is not realistic on CPU for production; use Colab GPU for inference.
- On Windows, `bitsandbytes` often fails; the service will still run, but may load without 4-bit.

## Colab + ngrok (recommended)

High-level steps:
1) In Colab, clone your repo that contains `qwen_adapter_files/`.
2) Install deps.
3) Run `uvicorn parser_service.app:app --host 0.0.0.0 --port 8000`.
4) Start ngrok pointing to port 8000.

Environment variables:
- `BASE_MODEL_ID` (default: `unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit`)
- `ADAPTER_PATH` (default: `../qwen_adapter_files` relative to `parser_service/`)
- `LOAD_IN_4BIT` (default: true)
- `MAX_NEW_TOKENS` (default: 256)

## Agent integration

In `tokaM107/langgraph_ai_agent`, replace the Gemini call in `app/services/llm.py` with an HTTP call to this service (example patch provided in the VS Code chat). 
