param(
  [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

Write-Host "Starting Qwen NLU Parser on port $Port" 

# Example:
#   $env:BASE_MODEL_ID = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"
#   $env:ADAPTER_PATH = "C:\\path\\to\\qwen_adapter_files"
#   $env:LOAD_IN_4BIT = "true"

python -m uvicorn parser_service.app:app --host 0.0.0.0 --port $Port
