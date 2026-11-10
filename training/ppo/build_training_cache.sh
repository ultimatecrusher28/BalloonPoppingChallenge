#!/bin/bash
# Rebuild EXACTLY the balloon caches the critic was trained on (seed list stored in
# critic_training_config_f1_g0_s7.json: train_seeds + holdout_seeds, all scenario 1).
# usage: training/ppo/build_training_cache.sh [n_parallel]   (≈20 s per seed per core; ~3224 seeds)
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-8}
PY=${PY:-../../.venv/bin/python}  # training/ppo/ (it ran as methods/v7/offline/)
$PY - <<'PY' > seeds.txt
import json
c = json.load(open("critic_training_config_f1_g0_s7.json"))
for s in c["train_seeds"] + c["holdout_seeds"]:
    print(int(s.replace("sc1_seed", "")))
PY
sort -un seeds.txt | xargs -P "$N" -I{} sh -c "$PY make_balloon_cache.py {} {} --out cache_screen > /dev/null 2>&1"
echo "cached: $(ls cache_screen | wc -l) files"
