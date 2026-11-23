#!/bin/bash
#SBATCH --job-name=space-glue-smoke
#SBATCH --partition=scc-cpu
#SBATCH --constraint=inet            # compute nodes only have internet access with this
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --qos=2h                     # priority boost for short jobs; remove if your account lacks it
#SBATCH --output=slurm-%x-%j.out
#
# CPU-only smoke test for the GWDG SCC. No GPU, no real model.
#
# It runs the complete SpaCE-GLUE workflow (dataset download, inference,
# scoring, aggregation) against a stub endpoint that always gives the same
# answer. Use it to check your setup before spending GPU hours, and to
# pre-download all datasets into data/. The scores are meaningless.
#
# Submit from the repository root:   sbatch slurm/scc_smoke_test.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "[$(date '+%F %T')] Smoke test starting on $HOSTNAME"

CONFIG="${CONFIG:-slurm/config.scc.yaml}"
RESULTS_DIR="results/smoke-test-$SLURM_JOB_ID"

if [ ! -x .venv/bin/space-glue ]; then
  echo "ERROR: .venv not found. Run 'bash slurm/setup_scc.sh' on a login node first" \
    "and submit this script from the repository root." >&2
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

# ---- start the stub model server on a free port ------------------------------
PORT=$(.venv/bin/python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
API_KEY="smoke-test"

.venv/bin/python slurm/stub_openai_server.py --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 60); do
  if curl -sf --noproxy '*' -H "Authorization: Bearer $API_KEY" \
    "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: the stub model server did not start." >&2
  exit 1
fi
echo "[$(date '+%F %T')] Model server ready on port $PORT"

# ---- run SpaCE-GLUE ----------------------------------------------------------
export SPACE_GLUE_MODEL="smoke-test-stub"
export SPACE_GLUE_BASE_URL="http://127.0.0.1:$PORT/v1"
export SPACE_GLUE_API_KEY="$API_KEY"
export SPACE_GLUE_RESULTS_DIR="$RESULTS_DIR"

RUN_LOG="$RESULTS_DIR/space-glue.log"
.venv/bin/space-glue --config "$CONFIG" 2>&1 | tee "$RUN_LOG"

# space-glue skips a dataset that fails and still exits with 0, so check the log.
echo
if grep -q "Error processing dataset" "$RUN_LOG"; then
  echo "SMOKE TEST FAILED - these datasets had errors:"
  grep "Error processing dataset" "$RUN_LOG"
  exit 1
fi
echo "Summaries written: $(find "$RESULTS_DIR" -name '*.summary.json' | wc -l)"
echo "SMOKE TEST PASSED - results (stub answers, not meaningful) are in $RESULTS_DIR"
