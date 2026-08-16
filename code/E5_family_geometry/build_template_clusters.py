#!/usr/bin/env python3
"""Build template-cluster identifiers for all joint16 perturbation pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REVISION_ROOT = Path(__file__).resolve().parents[1]
E5_ROOT = Path(__file__).resolve().parent
for p in (REVISION_ROOT, E5_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.path_registry import load_manifest, pairs_path  # noqa: E402
from template_cluster_utils import (  # noqa: E402
    infer_pair_template,
    lexical_stats,
    stable_template_id,
)


def load_pairs(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["families"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cluster_quality(cluster_sizes: dict[str, int]) -> dict[str, float]:
    sizes = list(cluster_sizes.values())
    n_pairs = sum(sizes)
    n_clusters = len(sizes)
    probs = [s / n_pairs for s in sizes] if n_pairs else []
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return {
        "singleton_cluster_fraction": sum(1 for s in sizes if s == 1) / max(1, n_clusters),
        "largest_cluster_size": max(sizes) if sizes else 0,
        "largest_cluster_fraction": max(sizes) / max(1, n_pairs) if sizes else 0.0,
        "median_cluster_size": float(sorted(sizes)[len(sizes) // 2]) if sizes else 0.0,
        "effective_n_clusters": math.exp(entropy) if sizes else 0.0,
        "cluster_size_entropy": entropy,
        "pairs_in_non_singleton_clusters": sum(s for s in sizes if s > 1),
        "non_singleton_pair_fraction": sum(s for s in sizes if s > 1) / max(1, n_pairs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build E5 template-cluster manifest")
    p.add_argument("--pairs-file", type=Path, default=None)
    p.add_argument(
        "--output-json",
        type=Path,
        default=E5_ROOT / "results" / "template_clusters.json",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=E5_ROOT / "results" / "template_clusters.md",
    )
    args = p.parse_args()

    manifest = load_manifest()
    pair_file = args.pairs_file or pairs_path(manifest)
    pair_sha = sha256_file(pair_file)
    families = load_pairs(pair_file)

    out: dict = {
        "pairs_file": str(pair_file),
        "pairs_sha256": pair_sha,
        "families": {},
    }
    md = [
        "# E5 Template Clusters",
        "",
        f"Pairs file: `{pair_file}`",
        "",
        f"Pairs SHA256: `{pair_sha}`",
        "",
        "| Family | pairs | templates | singleton clusters | largest cluster frac | effective clusters | unique values |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for family, block in sorted(families.items()):
        records = []
        template_to_id: dict[str, str] = {}
        by_template: dict[str, list[int]] = defaultdict(list)
        values: list[str] = []

        for idx, (orig, pert) in enumerate(block["pairs"]):
            inferred = infer_pair_template(orig, pert)
            template_id = template_to_id.setdefault(
                inferred.template_signature,
                stable_template_id(family, inferred.template_signature),
            )
            by_template[template_id].append(idx)
            values.extend([inferred.orig_value, inferred.pert_value])
            records.append(
                {
                    "pair_index": idx,
                    "template_id": template_id,
                    "template_signature": inferred.template_signature,
                    "orig_value": inferred.orig_value,
                    "pert_value": inferred.pert_value,
                    "diff_kind": inferred.diff_kind,
                    "orig": orig,
                    "pert": pert,
                }
            )

        lex = lexical_stats(values)
        sizes = {k: len(v) for k, v in sorted(by_template.items())}
        quality = cluster_quality(sizes)
        examples = []
        for template_id, member_indices in sorted(by_template.items())[:5]:
            first = records[member_indices[0]]
            examples.append(
                {
                    "template_id": template_id,
                    "cluster_size": len(member_indices),
                    "template_signature": first["template_signature"],
                    "example_orig": first["orig"],
                    "example_pert": first["pert"],
                }
            )
        out["families"][family] = {
            "n_pairs": len(records),
            "n_template_clusters": len(by_template),
            "template_cluster_sizes": sizes,
            "cluster_quality": quality,
            "representative_examples": examples,
            "lexical_value_stats": lex,
            "pairs": records,
        }
        md.append(
            f"| {family} | {len(records)} | {len(by_template)} | "
            f"{quality['singleton_cluster_fraction']:.3f} | "
            f"{quality['largest_cluster_fraction']:.3f} | "
            f"{quality['effective_n_clusters']:.2f} | {lex['unique_values']:.0f} |"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved -> {args.output_json}")
    print(f"Saved -> {args.output_md}")


if __name__ == "__main__":
    main()
