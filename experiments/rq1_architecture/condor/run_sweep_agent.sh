#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 SCRIPT PROJECT SWEEP_ID COUNT [EXTRA_ARGS...]" >&2
    exit 2
fi

export USER=rudo00002
export LOGNAME=rudo00002
export HOME=/home/rudo00002

PROJECT_DIR="$HOME/bachelor_thesis_RL/bachelors_thesis_ruslan"
WORK_DIR="${TMPDIR:-/tmp/$USER/bachelors_thesis_ruslan}"
SCRIPT="$1"
PROJECT="$2"
SWEEP_ID="$3"
COUNT="$4"
shift 4

mkdir -p "$WORK_DIR"

export PYTHONUSERBASE="$WORK_DIR/python-userbase"
export XDG_CACHE_HOME="$WORK_DIR/cache"
export UV_CACHE_DIR="$WORK_DIR/cache/uv"
export TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/torch_cache"
export MPLCONFIGDIR="$WORK_DIR/matplotlib"
export UV_PROJECT_ENVIRONMENT="$WORK_DIR/.venv"
export WANDB_DATA_DIR="$WORK_DIR/wandb_staging"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy

mkdir -p \
  "$PYTHONUSERBASE" \
  "$XDG_CACHE_HOME" \
  "$UV_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$MPLCONFIGDIR" \
  "$UV_PROJECT_ENVIRONMENT" \
  "$WANDB_DATA_DIR"

if [[ ! -x "$PYTHONUSERBASE/bin/uv" ]]; then
    python3 -m pip install --user --quiet uv
fi

cd "$PROJECT_DIR"
uv sync --quiet

if [[ -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
    echo "WANDB_API_KEY is not set and $HOME/.netrc does not exist." >&2
    exit 3
fi

uv run python "$SCRIPT" \
  --project "$PROJECT" \
  --sweep-id "$SWEEP_ID" \
  --count "$COUNT" \
  --device cpu \
  "$@"
