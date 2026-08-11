"""Multi-seed robustness analysis (paper R2-1).

Re-runs the meta-learning pipeline for every sample-size scale
(16/32/64/128/256/512) under several independent random seeds, reusing the
already-generated ROBIN score dumps (no GPU / no regeneration needed).

For each scale and held-out test attack (rotation, noise) it reports the
mean +/- std of the detector AUC and the per-seed recovery step k*, so the
scaling curve's seed-stability -- and whether the N=512 rotation point is a
genuine outlier or within normal seed variance -- can be assessed.

Prerequisite: run the pipeline once per scale (see README, "Run the full
pipeline") so that ``outputs_mwdras_<N>img/`` contains the meta config
(``meta_config_fomaml.json``) and the per-attack score dumps it references.

    python mwdras_multiseed.py
    python mwdras_multiseed.py --scales 16 32 512 --seeds 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mwdras_meta_runner as runner

REPO_ROOT = Path(__file__).resolve().parent

# (global_seed, split_seed) pairs. Index 0 reproduces the originally reported run.
SEED_PAIRS: List[tuple] = [
    (123, 456), (231, 654), (321, 789), (987, 147),
    (555, 999), (271, 314), (101, 202), (424, 838),
]

TEST_ATTACKS = ("robin_rotation", "robin_noise")


def _scale_dir(scale: int) -> Path:
    return REPO_ROOT / f"outputs_mwdras_{scale}img"


def _find_config(scale: int) -> Optional[Path]:
    """Locate the meta-learning config produced by the bridge for this scale."""
    d = _scale_dir(scale)
    for name in ("meta_config_fomaml.json", "paper4_meta_config_fomaml.json"):
        if (d / name).is_file():
            return d / name
    hits = sorted(d.glob("*meta_config*.json"))
    return hits[0] if hits else None


def _find_results(out_dir: Path) -> Path:
    """Return the newest ``*_results.json`` written under out_dir by the runner."""
    cands = sorted(out_dir.rglob("*_results.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"no *_results.json produced under {out_dir}")
    return cands[-1]


def _run_single(base_cfg: Dict[str, Any], scale: int, global_seed: int, split_seed: int) -> Dict[str, Any]:
    """Run one meta-learning fit with the given seeds and return its results dict."""
    out_dir = _scale_dir(scale) / "meta_multiseed" / f"seed_{global_seed}_{split_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(base_cfg)
    cfg["global_seed"] = global_seed
    cfg["split_seed"] = split_seed
    cfg["output_root"] = str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(cfg, tf)
        tmp_cfg_path = Path(tf.name)
    try:
        runner.run(tmp_cfg_path)
    finally:
        tmp_cfg_path.unlink(missing_ok=True)

    return json.loads(_find_results(out_dir).read_text(encoding="utf-8"))


def _extract_per_attack(results: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    rows = results["paper_questions"]["q1_all_baselines"]["task_rows"]
    return {
        r["task_id"]: {"k_star_meta": r.get("k_star_meta"), "auc_meta_max_k": r.get("auc_meta_max_k")}
        for r in rows
    }


def _aggregate(per_seed: List[Dict[str, Dict[str, Optional[float]]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for attack in TEST_ATTACKS:
        aucs = [s[attack]["auc_meta_max_k"] for s in per_seed if attack in s and s[attack]["auc_meta_max_k"] is not None]
        kstars = [s[attack]["k_star_meta"] for s in per_seed if attack in s]
        recovered = [k for k in kstars if k is not None]
        summary[attack] = {
            "n_seeds": len(per_seed),
            "auc_mean": round(statistics.mean(aucs), 4) if aucs else None,
            "auc_std": round(statistics.pstdev(aucs), 4) if len(aucs) > 1 else 0.0,
            "auc_min": round(min(aucs), 4) if aucs else None,
            "auc_max": round(max(aucs), 4) if aucs else None,
            "auc_values": [round(a, 4) for a in aucs],
            "recovery_rate": round(len(recovered) / len(kstars), 3) if kstars else None,
            "k_star_mean": round(statistics.mean(recovered), 3) if recovered else None,
            "k_star_values": kstars,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed robustness analysis (R2-1)")
    parser.add_argument("--scales", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--seeds", type=int, default=5, help="number of seed pairs to use (max %d)" % len(SEED_PAIRS))
    args = parser.parse_args()

    seed_pairs = SEED_PAIRS[:min(args.seeds, len(SEED_PAIRS))]
    overall: Dict[str, Any] = {"seed_pairs": seed_pairs, "test_attacks": list(TEST_ATTACKS), "scales": {}}

    for scale in args.scales:
        cfg_path = _find_config(scale)
        if cfg_path is None:
            print(f"[SKIP] scale {scale}: meta config not found under {_scale_dir(scale)}")
            continue
        base_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        print(f"\n{'='*60}\n[SCALE {scale}] running {len(seed_pairs)} seeds\n{'='*60}")
        per_seed: List[Dict[str, Dict[str, Optional[float]]]] = []
        for (gs, ss) in seed_pairs:
            print(f"  -> scale {scale} seed (global={gs}, split={ss})")
            per_seed.append(_extract_per_attack(_run_single(base_cfg, scale, gs, ss)))
        overall["scales"][str(scale)] = _aggregate(per_seed)

    out_path = REPO_ROOT / "MULTISEED_RESULTS.json"
    out_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print("\nwrote", out_path)

    print(f"\n{'='*70}\nMULTI-SEED SCALING SUMMARY\n{'='*70}")
    for attack in TEST_ATTACKS:
        print(f"\n### {attack}")
        print("| N | AUC mean +/- std | AUC range | recovery rate | mean k* |")
        print("|---|------------------|-----------|---------------|---------|")
        for scale in args.scales:
            s = overall["scales"].get(str(scale))
            if not s:
                continue
            a = s[attack]
            rng = f"{a['auc_min']}-{a['auc_max']}" if a["auc_min"] is not None else "-"
            print(f"| {scale} | {a['auc_mean']} +/- {a['auc_std']} | {rng} | {a['recovery_rate']} | {a['k_star_mean']} |")


if __name__ == "__main__":
    main()
