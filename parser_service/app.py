from __future__ import annotations

import os
import time

import torch
from fastapi import FastAPI, HTTPException

from .json_extract import extract_json_object
from .model_loader import load_model
from .prompt import SYSTEM_PROMPT
from .schemas import HealthResponse, ParseRequest, ParseResponse


app = FastAPI(title="Qwen NLU Parser", version="0.1.0")

_LOADED = None


def _get_loaded():
    global _LOADED
    if _LOADED is None:
        _LOADED = load_model()
    return _LOADED


def _coerce_response(obj: dict) -> ParseResponse:
    # Ensure required keys exist; keep MVP strict.
    intent = obj.get("intent")
    if intent not in {"routing", "general_info", "other"}:
        # If model returns unexpected intent, downgrade to other.
        intent = "other"

    constraints = obj.get("constraints")
    if not isinstance(constraints, list):
        constraints = []

    # Normalize empty strings to None
    def norm_str(v):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return None

    return ParseResponse(
        intent=intent,
        origin=norm_str(obj.get("origin")),
        destination=norm_str(obj.get("destination")),
        mode=norm_str(obj.get("mode")),
        constraints=[str(x) for x in constraints if str(x).strip()],
        query_type=norm_str(obj.get("query_type")),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _get_loaded()
    return HealthResponse(
        model=loaded.base_model_id,
        adapter_path=loaded.adapter_path,
        device=loaded.device,
        extra={
            "load_in_4bit": os.getenv("LOAD_IN_4BIT", "true"),
        },
    )


@app.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest) -> ParseResponse:
    loaded = _get_loaded()

    # Build a chat-style prompt compatible with Qwen templates.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.query},
    ]

    tokenizer = loaded.tokenizer
    model = loaded.model

    try:
        # Tokenize with chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt")
        else:
            # Fallback: simple concatenation
            text = SYSTEM_PROMPT + "\n\nUser:\n" + req.query + "\n\nJSON:\n"
            inputs = tokenizer(text, return_tensors="pt")

        # Move inputs to model device
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
                do_sample=False,
                temperature=0.0,
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )
        _ = time.time() - t0

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)

        obj = extract_json_object(decoded)
        return _coerce_response(obj)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"parse_failed: {e}")
