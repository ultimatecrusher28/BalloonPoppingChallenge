#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "usage: bash $0 <seed>" >&2
  exit 2
fi
SEED=$1

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CFG=finals/sc40_finals_lm10.yaml
SCN=BalloonPoppingGymEnv/envs/scenario_parameters/scenario_40_parameters.yaml

exec 9<"$ROOT/finals/run_finals.sh"
flock 9

BAK=$(mktemp)
cp -p "$SCN" "$BAK"
restore() { cp -p "$BAK" "$SCN"; rm -f "$BAK"; }
trap restore EXIT
trap 'exit 130' INT TERM HUP

sed -i -E "0,/^  random_seed:/s/^(  random_seed:)[[:space:]]*[^[:space:]#]+/\1 $SEED/" "$SCN"
grep -q -E "^  random_seed: $SEED( |$)" "$SCN" || { echo "failed to set random_seed" >&2; exit 1; }

[[ -x "$ROOT/.venv/bin/python" ]] || uv sync --frozen

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "seed $SEED"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS"
grep -E "^  (random_seed|date):" "$SCN"

RUN=("$ROOT/.venv/bin/python" BalloonPoppingGymEnv/evaluation/evaluate.py "$CFG")
if command -v srun >/dev/null 2>&1; then
  env srun -w art2 -p gpu -N 1 -n 1 -c 24 --mem=16G -t 1:00:00 -J "bpc_finals_$SEED" "${RUN[@]}"
else
  "${RUN[@]}"
fi
