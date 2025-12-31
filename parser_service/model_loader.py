from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass(frozen=True)
class LoadedModel:
    tokenizer: object
    model: object
    base_model_id: str
    adapter_path: str
    device: str


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_model() -> LoadedModel:
    """Load base model + LoRA adapter.

    Defaults match your adapter_config.json (Unsloth 4bit base).

    Env vars:
      - BASE_MODEL_ID
      - ADAPTER_PATH
      - LOAD_IN_4BIT (default true)
      - DEVICE (optional; if unset uses transformers device_map=auto)
    """

    base_model_id = os.getenv(
        "BASE_MODEL_ID", "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"
    )
    adapter_path = os.getenv("ADAPTER_PATH")
    if not adapter_path:
        # default: ../qwen_adapter_files relative to this file
        adapter_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "qwen_adapter_files")
        )

    load_in_4bit = _env_bool("LOAD_IN_4BIT", True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    # Model
    # Note: on Windows, bitsandbytes is often unavailable; allow CPU fallback.
    device_override = os.getenv("DEVICE")  # e.g. "cuda" or "cpu"

    model_kwargs: dict = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }

    if device_override:
        # If user forces a device, don't use device_map=auto
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = "auto"

    # 4-bit quantization is only meaningful on CUDA with bitsandbytes
    if load_in_4bit and torch.cuda.is_available():
        model_kwargs["load_in_4bit"] = True

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    if device_override:
        base_model = base_model.to(device_override)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    if device_override:
        device = device_override
    else:
        # best-effort reporting
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return LoadedModel(
        tokenizer=tokenizer,
        model=model,
        base_model_id=base_model_id,
        adapter_path=adapter_path,
        device=device,
    )
