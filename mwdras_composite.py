"""Composite-attack scaling analysis (paper R2-2).

Evaluates the meta-initialized detector -- meta-trained only on the SINGLE
distortions (none / jpeg / blurring / color_jitter, validated on cropping) --
against the held-out COMPOSITE attacks rotation+jpeg and rotation+cropping at
every sample scale, reusing already-generated composite ROBIN score dumps
(no GPU, no regeneration).

This mirrors the realistic deployment scenario of Reviewer 2-2: a detector
whose meta-initialization only ever saw *single* distortions must recover on
*compound* distortions it never trained on. For each scale and composite
attack it reports the mean +/- std (over several seeds) of the zero-shot and
meta-recovered AUC and the recovery step k*.

Prerequisites (see README):
  1. Run the pipeline once per scale so outputs_mwdras_<N>img/ has the meta
     config and per-attack single-distortion task files (tasks/robin_*.json).
  2. Generate the composite score dump per scale with the bridge's composite
     mode (python mwdras_bridge.py --composite ...); point --combo-dump at it.

    python mwdras_composite.py
    python mwdras_composite.py --scales 16 512 --seeds 5 \
        --combo-dump outputs_mwdras_{scale}img_composite/robin_runtime_score_dump.json
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

# Composite held-out test tasks: task_id -> attack_key inside the combo dump.
COMPOSITE_TASKS: Dict[str, str] = {
    "robin_rotation_jpeg": "rotation+jpeg",
    "robin_rotation_cropping": "rotation+cropping",
}
TEST_ATTACKS = tuple(COMPOSITE_TASKS.keys())

# Meta-training stays on the single-attack substrate (unchanged from the main paper).
SINGLE_TRAIN = ["robin_none", "robin_jpeg", "robin_blurring", "robin_color_jitter"]
SINGLE_VAL = ["robin_cropping"]
CALIB_TASK_ID = "robin_none"

# Default per-scale composite score dump ({scale} is substituted).
DEFAULT_COMBO_DUMP = "outputs_mwdras_{scale}img_composite/robin_runtime_score_dump.json"


def _abs(p: Path) -> str:
    return p.resolve().as_posix()


def _scale_dir(scale: int) -> Path:
    return REPO_ROOT / f"outputs_mwdras_{scale}img"


def _find_config(scale: int) -> Optional[Path]:
    d = _scale_dir(scale)
    for name in ("meta_config_fomaml.json", "paper4_meta_config_fomaml.json"):
        if (d / name).is_file():
            return d / name
    hits = sorted(d.glob("*meta_config*.json"))
    return hits[0] if hits else None


def _combo_dump(scale: int, template: str) -> Path:
    return (REPO_ROOT / template.format(scale=scale)).resolve()


def _single_task_file(scale: int, task_id: str) -> Path:
    return _scale_dir(scale) / "tasks" / f"{task_id}.json"


def _find_results(out_dir: Path) -> Path:
    cands = sorted(out_dir.rglob("*_results.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"no *_results.json produced under {out_dir}")
    return cands[-1]


def _build_manifest(scale: int, combo: Path) -> Dict[str, Any]:
    """Single-attack train/val tasks (existing dumps) plus the two composite test
    tasks pointing at the combo dump via attack_key."""
    tasks: List[Dict[str, Any]] = []
    for tid in SINGLE_TRAIN + SINGLE_VAL:
        tasks.append({"task_id": tid, "score_json": _abs(_single_task_file(scale, tid))})
    for tid, attack_key in COMPOSITE_TASKS.items():
        tasks.append({"task_id": tid, "score_json": _abs(combo), "attack_key": attack_key})
    return {"calibration_task_id": CALIB_TASK_ID, "tasks": tasks}


def _run_single(base_cfg: Dict[str, Any], scale: int, global_seed: int, split_seed: int, combo: Path) -> Dict[str, Any]:
    work = _scale_dir(scale) / "composite_meta" / f"seed_{global_seed}_{split_seed}"
    (work / "out").mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(scale, combo)
    # Write the manifest under both known runner conventions for robustness.
    for mname in ("task_manifest.json", "paper4_task_manifest.json"):
        (work / mname).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cfg = dict(base_cfg)
    cfg["global_seed"] = global_seed
    cfg["split_seed"] = split_seed
    cfg["data_root"] = _abs(work)
    cfg["output_root"] = _abs(work / "out")
    cfg["meta_train_task_ids"] = list(SINGLE_TRAIN)
    cfg["meta_val_task_ids"] = list(SINGLE_VAL)
    cfg["meta_test_task_ids"] = list(COMPOSITE_TASKS.keys())

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(cfg, tf)
        tmp_cfg_path = Path(tf.name)
    try:
        runner.run(tmp_cfg_path)
    finally:
        tmp_cfg_path.unlink(missing_ok=True)

    return json.loads(_find_results(work / "out").read_text(encoding="utf-8"))


def _extract_per_attack(results: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    """Recovered (max-k) AUC + k_star from q1; zero-shot (k=0) AUC from q3."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for row in results["paper_questions"]["q1_all_baselines"]["task_rows"]:
        out[row["task_id"]] = {"k_star_meta": row.get("k_star_meta"), "auc_meta_max_k": row.get("auc_meta_max_k")}
    for row in results["paper_questions"]["q3_amortized_verifier_tradeoff"]["task_rows"]:
        if row["task_id"] in out:
            out[row["task_id"]]["auc_zero_shot"] = row.get("auc_amortized")
    return out


def _aggregate(per_seed: List[Dict[str, Dict[str, Optional[float]]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for attack in TEST_ATTACKS:
        aucs = [s[attack]["auc_meta_max_k"] for s in per_seed if attack in s and s[attack]["auc_meta_max_k"] is not None]
        zs = [s[attack].get("auc_zero_shot") for s in per_seed if attack in s and s[attack].get("auc_zero_shot") is not None]
        kstars = [s[attack]["k_star_meta"] for s in per_seed if attack in s]
        recovered = [k for k in kstars if k is not None]
        summary[attack] = {
            "n_seeds": len(per_seed),
            "auc_mean": round(statistics.mean(aucs), 4) if aucs else None,
            "auc_std": round(statistics.pstdev(aucs), 4) if len(aucs) > 1 else 0.0,
            "auc_min": round(min(aucs), 4) if aucs else None,
            "auc_max": round(max(aucs), 4) if aucs else None,
            "auc_values": [round(a, 4) for a in aucs],
            "zero_shot_auc_mean": round(statistics.mean(zs), 4) if zs else None,
            "recovery_rate": round(len(recovered) / len(kstars), 3) if kstars else None,
            "k_star_mean": round(statistics.mean(recovered), 3) if recovered else None,
            "k_star_values": kstars,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite-attack scaling analysis (R2-2)")
    parser.add_argument("--scales", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--seeds", type=int, default=5, help="number of seed pairs to use (max %d)" % len(SEED_PAIRS))
    parser.add_argument("--combo-dump", default=DEFAULT_COMBO_DUMP,
                        help="path template to the composite score dump; {scale} is substituted")
    args = parser.parse_args()

    seed_pairs = SEED_PAIRS[:min(args.seeds, len(SEED_PAIRS))]
    overall: Dict[str, Any] = {
        "description": "R2-2 composite-attack scaling: meta-trained on single attacks, "
                       "evaluated on held-out composite attacks (rotation+jpeg, rotation+cropping).",
        "seed_pairs": seed_pairs,
        "test_attacks": list(TEST_ATTACKS),
        "scales": {},
    }

    for scale in args.scales:
        cfg_path = _find_config(scale)
        if cfg_path is None:
            print(f"[SKIP] scale {scale}: meta config not found under {_scale_dir(scale)}")
            continue
        combo = _combo_dump(scale, args.combo_dump)
        if not combo.is_file():
            print(f"[SKIP] scale {scale}: composite dump not found at {combo}")
            continue
        base_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        print(f"\n{'='*60}\n[SCALE {scale}] running {len(seed_pairs)} seeds (composite)\n{'='*60}")
        per_seed: List[Dict[str, Dict[str, Optional[float]]]] = []
        for (gs, ss) in seed_pairs:
            print(f"  -> scale {scale} seed (global={gs}, split={ss})")
            per_seed.append(_extract_per_attack(_run_single(base_cfg, scale, gs, ss, combo)))
        overall["scales"][str(scale)] = _aggregate(per_seed)

    out_path = REPO_ROOT / "COMPOSITE_RESULTS.json"
    out_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print("\nwrote", out_path)

    print(f"\n{'='*78}\nCOMPOSITE-ATTACK SCALING SUMMARY\n{'='*78}")
    for attack in TEST_ATTACKS:
        print(f"\n### {attack}")
        print("| N | zero-shot AUC | recovered AUC mean | AUC std | recovery rate | mean k* |")
        print("|---|---------------|--------------------|---------|---------------|---------|")
        for scale in args.scales:
            s = overall["scales"].get(str(scale))
            if not s:
                continue
            a = s[attack]
            print(f"| {scale} | {a['zero_shot_auc_mean']} | {a['auc_mean']} | {a['auc_std']} | {a['recovery_rate']} | {a['k_star_mean']} |")


if __name__ == "__main__":
    main()
