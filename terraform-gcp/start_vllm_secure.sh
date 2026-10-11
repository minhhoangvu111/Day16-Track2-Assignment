#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-google/gemma-4-E2B-it}"
SECRET_DIR="/dev/shm/vllm-secrets"
SECRET_FILE="${SECRET_DIR}/hf-token"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed yet; wait for the startup script to finish." >&2
  exit 1
fi
if ! sudo docker info >/dev/null 2>&1; then
  echo "Docker is not ready; wait for the startup script to finish." >&2
  exit 1
fi

read -r -s -p "Paste Hugging Face read token (input hidden): " HF_TOKEN
printf '\n'
if [[ -z "${HF_TOKEN}" ]]; then
  echo "Token cannot be empty." >&2
  exit 1
fi

sudo install -d -m 700 "${SECRET_DIR}"
printf '%s' "${HF_TOKEN}" | sudo tee "${SECRET_FILE}" >/dev/null
unset HF_TOKEN
sudo chmod 600 "${SECRET_FILE}"

sudo docker rm -f vllm >/dev/null 2>&1 || true
sudo docker run -d \
  --name vllm \
  --gpus all \
  -p 8000:8000 \
  --mount "type=bind,src=${SECRET_FILE},dst=/run/secrets/hf-token,readonly" \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest \
  -lc 'export HUGGING_FACE_HUB_TOKEN="$(cat /run/secrets/hf-token)"; exec vllm serve "$1" --dtype half --max-model-len 4096 --host 0.0.0.0' \
  -- "${MODEL_ID}"

echo "vLLM container started. The token is held in VM memory, not Terraform state or instance metadata."
echo "Check logs with: sudo docker logs -f vllm"
