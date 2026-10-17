#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime=${VIM_V2_PYTHON:-/root/miniconda3/envs/clip/bin/python}
"$runtime" - "$project/configs/vision_mamba_clip/v2/extensions.disabled.json" "${1:-}" <<'PY'
import json,sys
items=json.load(open(sys.argv[1])); selected=[r for r in items if not sys.argv[2] or r['name']==sys.argv[2]]
if not selected: raise SystemExit('Unknown extension')
for item in selected: print(json.dumps(item,indent=2))
raise SystemExit('Extensions are disabled. No training was started; a separately reviewed protocol is required.')
PY
