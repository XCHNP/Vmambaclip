#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
plan="$project/configs/vision_mamba_clip/v2/plan.json"
bash "$project/scripts/vim_v2.sh" budget --plan "$plan"
bash "$project/scripts/vim_v2.sh" lock-plan --plan "$plan"
bash "$project/scripts/vim_v2.sh" run-plan --plan "$plan" --group core --resume-incomplete
