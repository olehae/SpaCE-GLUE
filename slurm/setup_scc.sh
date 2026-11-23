#!/bin/bash
# One-time setup of the SpaCE-GLUE Python environment on the GWDG SCC.
# Run it on a login node (they have internet access):
#
#     bash slurm/setup_scc.sh
#
# It creates .venv/ in the repository root with Python 3.13 and installs
# SpaCE-GLUE into it. Running it again only updates the installation.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  module load uv
fi

# The uv cache lives in $HOME; if the repository is on another filesystem
# (e.g. $PROJECT) uv has to copy instead of hardlink. This silences the warning.
export UV_LINK_MODE=copy

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.13 .venv
fi
uv pip install --python .venv/bin/python -e .

# Datasets are cached as data/<name>_data.jsonl relative to the working
# directory; the directory has to exist before the first run.
mkdir -p data

.venv/bin/space-glue --help >/dev/null
echo
echo "Setup complete. Next step: sbatch slurm/scc_smoke_test.sbatch"
