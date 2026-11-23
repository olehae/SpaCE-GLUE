#!/bin/bash
#SBATCH --job-name=space-glue
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1                    # one 80 GB A100; use e.g. H100:1 or A100:2 for bigger models
#SBATCH --constraint=inet            # compute nodes only have internet access with this
#SBATCH --time=12:00:00              # maximum on scc-gpu is 48:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Runs SpaCE-GLUE on the GWDG SCC: starts a vLLM server for $MODEL on the
# allocated GPU(s) and evaluates it through the OpenAI-compatible API.
#
# Submit from the repository root (see slurm/README.md):
#
#     sbatch slurm/scc_vllm.sbatch
#     sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-8B-Instruct slurm/scc_vllm.sbatch
#
# CPUs and memory are left to Slurm, which assigns a fair share per GPU.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "[$(date '+%F %T')] Job starting on $HOSTNAME"

# ---- settings (override with `sbatch --export=ALL,NAME=value ...`) ------------
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"   # Hugging Face repo id or local model directory
MODEL_NAME="${MODEL_NAME:-$MODEL}"           # name used in the API and in the result file names
CONFIG="${CONFIG:-slurm/config.scc.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results}"
VLLM_SIF="${VLLM_SIF:-$PWD/vllm-openai.sif}" # Apptainer image that contains vLLM
VLLM_ARGS="${VLLM_ARGS:-}"                   # extra arguments for `vllm serve`

if [ ! -x .venv/bin/space-glue ]; then
  echo "ERROR: .venv not found. Run 'bash slurm/setup_scc.sh' on a login node first" \
    "and submit this script from the repository root." >&2
  exit 1
fi
if [ ! -f "$VLLM_SIF" ]; then
  echo "ERROR: vLLM image not found at $VLLM_SIF." \
    "Set VLLM_SIF to an Apptainer image with vLLM (see slurm/README.md)." >&2
  exit 1
fi

# Keep the Hugging Face cache out of $HOME (small quota).
if [ -z "${HF_HOME:-}" ]; then
  if [ -n "${PROJECT:-}" ]; then
    export HF_HOME="$PROJECT/$USER/hf"
  else
    export HF_HOME="$HOME/.cache/huggingface"
    echo "WARNING: HF_HOME and PROJECT are unset, using $HF_HOME" >&2
  fi
fi
echo "Model: $MODEL | image: $VLLM_SIF | HF_HOME: $HF_HOME"

# The uv-managed Python does not find the system CA certificates on its own;
# without this, dataset downloads fail with CERTIFICATE_VERIFY_FAILED.
if [ -z "${SSL_CERT_FILE:-}" ] && [ -f /etc/pki/tls/certs/ca-bundle.crt ]; then
  export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
fi

# `--constraint=inet` routes http(s) through the GWDG proxy. Requests to the
# local model server must not go through it.
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
export NO_PROXY="$no_proxy"

# Datasets are cached in data/ relative to the working directory. A failed
# download can leave an empty file behind, which would be read as an empty
# dataset on the next run.
mkdir -p data "$RESULTS_DIR" "$HF_HOME"
find data -maxdepth 1 -name '*_data.jsonl' -size 0 -delete

# ---- start the vLLM server on a free port ------------------------------------
PORT=$(.venv/bin/python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
# GPU nodes are shared: protect the server with a random key.
API_KEY=$(.venv/bin/python -c 'import secrets; print(secrets.token_hex(16))')
VLLM_LOG="$PWD/slurm-vllm-$SLURM_JOB_ID.log"

module load gcc cuda apptainer

# The container starts with a clean environment; APPTAINERENV_* variables are
# passed into it.
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_VLLM_API_KEY="$API_KEY"
if [ -n "${HF_TOKEN:-}" ]; then
  export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
fi
for var in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
  if [ -n "${!var:-}" ]; then
    export "APPTAINERENV_$var=${!var}"
  fi
done

BINDS=(-B "$HF_HOME:$HF_HOME:rw")
if [ -d "$MODEL" ]; then
  MODEL=$(realpath "$MODEL")
  BINDS+=(-B "$MODEL:$MODEL:ro")
fi

# shellcheck disable=SC2086  # VLLM_ARGS is split into separate arguments on purpose
apptainer exec --nv --cleanenv "${BINDS[@]}" "$VLLM_SIF" \
  vllm serve "$MODEL" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size "${SLURM_GPUS_ON_NODE:-1}" \
  $VLLM_ARGS \
  >"$VLLM_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# Downloading and loading the model can take a while: wait up to one hour.
echo "[$(date '+%F %T')] Waiting for vLLM (log: $VLLM_LOG)"
READY=0
for _ in $(seq 1 720); do
  if curl -sf --noproxy '*' -H "Authorization: Bearer $API_KEY" \
    "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: vLLM did not become ready. Last lines of $VLLM_LOG:" >&2
  tail -n 40 "$VLLM_LOG" >&2
  exit 1
fi
echo "[$(date '+%F %T')] Model server ready on port $PORT"

# ---- run SpaCE-GLUE ----------------------------------------------------------
export SPACE_GLUE_MODEL="$MODEL_NAME"
export SPACE_GLUE_BASE_URL="http://127.0.0.1:$PORT/v1"
export SPACE_GLUE_API_KEY="$API_KEY"
export SPACE_GLUE_RESULTS_DIR="$RESULTS_DIR"

RUN_LOG="$RESULTS_DIR/space-glue-$SLURM_JOB_ID.log"
.venv/bin/space-glue --config "$CONFIG" 2>&1 | tee "$RUN_LOG"

# space-glue skips a dataset that fails and still exits with 0, so check the log.
echo
if grep -q "Error processing dataset" "$RUN_LOG"; then
  echo "FINISHED WITH ERRORS - these datasets had errors:"
  grep "Error processing dataset" "$RUN_LOG"
  exit 1
fi
echo "[$(date '+%F %T')] Done. Results are in $RESULTS_DIR"
