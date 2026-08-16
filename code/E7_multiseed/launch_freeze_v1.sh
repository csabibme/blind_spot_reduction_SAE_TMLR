#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate
export SAE_REPO_ROOT="$(cd ../.. && pwd)"

PAIRS="FINAL/joint16_experiment/data/joint16_pairs.json"
RUN_ROOT="E7_multiseed/runs/freeze_v1"

python - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

root = Path.cwd()
manifest = root / "manifest.yaml"
freeze = root / "E7_multiseed" / "E7_PROTOCOL_FREEZE.md"
out = root / "E7_multiseed" / "results" / "protocol_manifest_snapshot.json"
out.parent.mkdir(parents=True, exist_ok=True)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "freeze_id": "E7_FREEZE_V1",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "manifest": str(manifest),
    "manifest_sha256": sha(manifest),
    "freeze_doc": str(freeze),
    "freeze_doc_sha256": sha(freeze),
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY

run_one() {
  local profile="$1"
  local seed_label="$2"
  local seed="$3"
  local objective="$4"
  local lambda_v="$5"
  local owt_cache="$6"
  local perturb_cache="$7"

  local run_name="e7_${profile}_${objective}_${seed_label}_freeze_v1"
  local output_dir="${RUN_ROOT}/${profile}/${seed_label}/${objective}"
  mkdir -p "$output_dir"

  echo "=== E7 ${profile} ${seed_label} ${objective} ==="
  python E7_multiseed/train_e7_joint_hybrid.py \
    --profile "$profile" \
    --objective "$objective" \
    --seed-label "$seed_label" \
    --seed "$seed" \
    --run-name "$run_name" \
    --output-dir "$output_dir" \
    --pairs-file "$SAE_REPO_ROOT/$PAIRS" \
    --owt-cache "$owt_cache" \
    --perturb-cache "$perturb_cache" \
    --lambda-v "$lambda_v" \
    --steps 15000 \
    --batch-size 64 \
    --l1-coeff 0.001 \
    --d-sae 4096 \
    --v-families-per-step 16 \
    --v-pairs-per-family 8 \
    --device mps \
    --dtype float16 \
    --log-every 500
}

run_profile() {
  local profile="$1"
  local owt_cache="$2"
  local perturb_cache="$3"

  run_one "$profile" seed_000 0 standard 0.0 "$owt_cache" "$perturb_cache"
  run_one "$profile" seed_000 0 vreg 0.2 "$owt_cache" "$perturb_cache"
  run_one "$profile" seed_001 1 standard 0.0 "$owt_cache" "$perturb_cache"
  run_one "$profile" seed_001 1 vreg 0.2 "$owt_cache" "$perturb_cache"
  run_one "$profile" seed_002 2 standard 0.0 "$owt_cache" "$perturb_cache"
  run_one "$profile" seed_002 2 vreg 0.2 "$owt_cache" "$perturb_cache"
}

run_profile \
  gpt2 \
  "../tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gpt2_l12_25k.pt" \
  "E7_multiseed/runs/freeze_v1/_caches/gpt2_true_last_perturb_cache.pt"

run_profile \
  gemma-2-2b \
  "../tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gemma_l13_25k.pt" \
  "E7_multiseed/runs/freeze_v1/_caches/gemma-2-2b_true_last_perturb_cache.pt"

run_profile \
  qwen-2.5-3b \
  "../tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_qwen_l18_25k.pt" \
  "E7_multiseed/runs/freeze_v1/_caches/qwen-2.5-3b_true_last_perturb_cache.pt"

echo "E7_FREEZE_V1 training complete."
