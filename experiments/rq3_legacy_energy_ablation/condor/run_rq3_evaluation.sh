#!/bin/bash
set -euo pipefail

export USER=rudo00002
export LOGNAME=rudo00002
export HOME=/home/rudo00002

PROJECT_DIR="$HOME/bachelor_thesis_RL/bachelors_thesis_ruslan"
WORK_DIR="${TMPDIR:-/tmp/$USER/bachelors_thesis_ruslan}"

export PYTHONUSERBASE="$WORK_DIR/python-userbase"
export XDG_CACHE_HOME="$WORK_DIR/cache"
export UV_CACHE_DIR="$WORK_DIR/cache/uv"
export TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/torch_cache"
export MPLCONFIGDIR="$WORK_DIR/matplotlib"
export UV_PROJECT_ENVIRONMENT="$WORK_DIR/.venv"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg

mkdir -p \
  "$PYTHONUSERBASE" \
  "$XDG_CACHE_HOME" \
  "$UV_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$MPLCONFIGDIR" \
  "$UV_PROJECT_ENVIRONMENT"

if [[ ! -x "$PYTHONUSERBASE/bin/uv" ]]; then
    python3 -m pip install --user --quiet uv
fi

cd "$PROJECT_DIR"
uv sync --quiet

uv run python minigrid_part/run_rq3_evaluation.py \
  --models-root experiments/rq3_legacy_energy_ablation/continuations \
  --output-root experiments/rq3_legacy_energy_ablation/evaluation \
  --episodes 100 \
  --dataset-episodes 30 \
  --dataset-samples 256 \
  --bootstrap-repeats 2000 \
  --batch-size 32 \
  --device cpu
