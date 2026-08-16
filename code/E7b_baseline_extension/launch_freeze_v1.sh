#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate
export SAE_REPO_ROOT="$(cd ../.. && pwd)"

PAIRS="FINAL/joint16_experiment/data/joint16_pairs.json"
RUN_ROOT="E7b_baseline_extension/runs/freeze_v1"
E7A_CACHE_ROOT="E7_multiseed/runs/freeze_v1/_caches"

python - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

root = Path.cwd()
out = root / "E7b_baseline_extension" / "results" / "protocol_manifest_snapshot.json"
out.parent.mkdir(parents=True, exist_ok=True)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "freeze_id": "E7B_BASELINE_FREEZE_V1",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "manifest": str(root / "manifest.yaml"),
    "manifest_sha256": sha(root / "manifest.yaml"),
    "freeze_doc": str(root / "E7b_baseline_extension" / "E7B_PROTOCOL_FREEZE.md"),
    "freeze_doc_sha256": sha(root / "E7b_baseline_extension" / "E7B_PROTOCOL_FREEZE.md"),
    "main_goal": str(root / "MAIN_rev_GOAL.md"),
    "main_goal_sha256": sha(root / "MAIN_rev_GOAL.md"),
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY

run_one() {
  local profile="$1"
  local seed_label="$2"
  local seed="$3"
  local objective="$4"
  local owt_cache="$5"
  local perturb_cache="$6"

  local run_name="e7b_${profile}_${objective}_${seed_label}_freeze_v1"
  local output_dir="${RUN_ROOT}/${profile}/${seed_label}/${objective}"
  mkdir -p "$output_dir"

  echo "=== E7b ${profile} ${seed_label} ${objective} ==="
  python E7b_baseline_extension/train_e7b_baseline.py \
    --profile "$profile" \
    --objective "$objective" \
    --seed-label "$seed_label" \
    --seed "$seed" \
    --run-name "$run_name" \
    --output-dir "$output_dir" \
    --pairs-file "$SAE_REPO_ROOT/$PAIRS" \
    --owt-cache "$owt_cache" \
    --perturb-cache "$perturb_cache" \
    --steps 15000 \
    --batch-size 64 \
    --l1-coeff 0.001 \
    --d-sae 4096 \
    --device mps \
    --dtype float16 \
    --log-every 500
}

run_profile() {
  local profile="$1"
  local owt_cache="$2"
  local perturb_cache="$3"

  for seed_label in seed_000 seed_001 seed_002; do
    local seed="${seed_label#seed_}"
    seed=$((10#$seed))
    run_one "$profile" "$seed_label" "$seed" jumprelu "$owt_cache" "$perturb_cache"
    run_one "$profile" "$seed_label" "$seed" mdl "$owt_cache" "$perturb_cache"
  done
}

run_profile \
  gpt2 \
  "../tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gpt2_l12_25k.pt" \
  "${E7A_CACHE_ROOT}/gpt2_true_last_perturb_cache.pt"

run_profile \
  gemma-2-2b \
  "../tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gemma_l13_25k.pt" \
  "${E7A_CACHE_ROOT}/gemma-2-2b_true_last_perturb_cache.pt"

run_profile \
  qwen-2.5-3b \
  "../tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_qwen_l18_25k.pt" \
  "${E7A_CACHE_ROOT}/qwen-2.5-3b_true_last_perturb_cache.pt"

echo "E7B_BASELINE_FREEZE_V1 training complete."
