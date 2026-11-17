#!/bin/bash
set -euo pipefail

BCOS_B="${1:-2.3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="$ROOT/experiments/rq1_architecture/created_sweep_ids.txt"

cd "$ROOT"
mkdir -p experiments/rq1_architecture/condor/logs
: > "$OUTPUT"

create_sweep() {
    local name="$1"
    shift
    local result
    result="$(uv run python "$@" --create-sweep)"
    printf '%s\n' "$result"
    local sweep_id
    sweep_id="$(printf '%s\n' "$result" | awk '/Created sweep:/ {print $3}' | tail -n 1)"
    if [[ -z "$sweep_id" ]]; then
        echo "Could not parse sweep ID for $name" >&2
        exit 4
    fi
    printf '%s=%s\n' "$name" "$sweep_id" >> "$OUTPUT"
}

create_sweep PPO_ARCHITECTURE \
    minigrid_part/sweep_rq1_ppo_architecture.py
create_sweep PPO_B_POWER \
    minigrid_part/sweep_rq1_ppo_b_power.py
create_sweep DQN_ARCHITECTURE \
    minigrid_part/sweep_rq1_dqn_architecture.py --bcos-b 2.3
create_sweep MINIGRID_SUPPORT \
    minigrid_part/sweep_rq1_minigrid_support.py --bcos-b "$BCOS_B"
create_sweep CARRACING_ARCHITECTURE \
    car_racing_part/sweep_rq1_carracing_architecture.py --bcos-b "$BCOS_B"

echo "Saved sweep IDs to: $OUTPUT"
