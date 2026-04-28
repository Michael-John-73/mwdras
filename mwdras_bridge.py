from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mwdras_meta_runner import run as run_meta


# Canonical single attacks
SINGLE_ATTACKS = ["rotation", "jpeg", "cropping", "blurring", "noise", "color_jitter"]

# Legacy 7-attack list (for backward compat with existing score dumps)
ATTACKS_7 = ["none"] + SINGLE_ATTACKS


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _generate_combo_attacks(max_severity: int = 6) -> List[Tuple[str, List[str], int]]:
    """Generate all subset-combo attacks up to max_severity.

    Returns list of (attack_key, attack_components, severity) tuples.
    attack_key uses '+' separator, e.g. 'rotation+noise'.
    severity = number of attacks in the combination.
    """
    combos: List[Tuple[str, List[str], int]] = []
    # severity 0: none
    combos.append(("none", [], 0))
    # severity 1..max_severity: subsets of SINGLE_ATTACKS
    for r in range(1, min(max_severity, len(SINGLE_ATTACKS)) + 1):
        for subset in itertools.combinations(SINGLE_ATTACKS, r):
            key = "+".join(subset)
            combos.append((key, list(subset), r))
    return combos


def _split_combo_tasks(
    combos: List[Tuple[str, List[str], int]],
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split combo tasks into train/val/test by severity.

    Strategy:
      - Train: none (s=0) + ALL single attacks (s=1) + SOME 2-combos (s=2)
      - Val:   remaining 2-combos
      - Test:  all 3+ combos (s>=3) — truly unseen complexity
    For 2-combos, 60% train / 40% val split (seeded).
    """
    import random
    rng = random.Random(seed)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    s2_combos: List[str] = []

    for key, _parts, severity in combos:
        task_id = f"robin_{key}"
        if severity <= 1:
            train_ids.append(task_id)
        elif severity == 2:
            s2_combos.append(task_id)
        else:  # severity >= 3
            test_ids.append(task_id)

    # Split s=2 combos: 60% train, 40% val
    rng.shuffle(s2_combos)
    n_train_s2 = max(1, int(len(s2_combos) * 0.6))
    train_ids.extend(s2_combos[:n_train_s2])
    val_ids.extend(s2_combos[n_train_s2:])

    return train_ids, val_ids, test_ids


def _run_robin_inject(args: argparse.Namespace, score_dump_json: Path, attack_keys: List[str] | None = None) -> None:
    prompts_path = Path(args.prompts_file)
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    if prompts_path.exists():
        lines = [ln.strip() for ln in prompts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        lines = []
    needed = max(0, int(args.end))
    if len(lines) < needed:
        filler = "a high quality photo of a mountain lake at sunrise"
        if not lines:
            lines = [filler]
        while len(lines) < needed:
            lines.append(lines[-1])
        prompts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd: List[str] = [
        sys.executable,
        str(Path(args.robin_root) / "inject_wm_inner_latent_robin.py"),
        "--run_name",
        args.run_name,
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--model_id",
        args.model_id,
        "--wm_path",
        str(Path(args.wm_path).resolve()),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--test_num_inference_steps",
        str(args.test_num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--w_channel",
        str(args.w_channel),
        "--w_pattern",
        args.w_pattern,
        "--w_mask_shape",
        args.w_mask_shape,
        "--w_up_radius",
        str(args.w_up_radius),
        "--w_low_radius",
        str(args.w_low_radius),
        "--prompts_file",
        str(Path(args.prompts_file).resolve()),
        "--dump_scores_json",
        str(score_dump_json.resolve()),
    ]

    if args.reference_model:
        cmd.extend(["--reference_model", args.reference_model])
    if args.reference_model_pretrain:
        cmd.extend(["--reference_model_pretrain", args.reference_model_pretrain])
    if attack_keys is not None:
        cmd.extend(["--attack_list", ",".join(attack_keys)])

    print("[RUN] ROBIN inject command:")
    print(" ".join(cmd))

    proc = subprocess.run(cmd, cwd=str(args.robin_root), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ROBIN injection failed with exit code {proc.returncode}")


def _run_ddim_step_sweep(args: argparse.Namespace, output_root: Path, step_counts: List[int]) -> None:
    """Run ROBIN at multiple test_num_inference_steps for Q3 DDIM tradeoff analysis."""
    ddim_dir = output_root / "ddim_steps"
    ddim_dir.mkdir(parents=True, exist_ok=True)

    for n_steps in step_counts:
        dump_file = ddim_dir / f"scores_step_{n_steps}.json"
        if dump_file.exists():
            print(f"[DDIM] step={n_steps} already exists, skipping")
            continue

        print(f"[DDIM] Running ROBIN with test_num_inference_steps={n_steps}")
        prompts_path = Path(args.prompts_file).resolve()

        cmd: List[str] = [
            sys.executable,
            str(Path(args.robin_root) / "inject_wm_inner_latent_robin.py"),
            "--run_name", f"ddim_step_{n_steps}",
            "--start", str(args.start),
            "--end", str(args.end),
            "--model_id", args.model_id,
            "--wm_path", str(Path(args.wm_path).resolve()),
            "--num_inference_steps", str(args.num_inference_steps),
            "--test_num_inference_steps", str(n_steps),
            "--guidance_scale", str(args.guidance_scale),
            "--w_channel", str(args.w_channel),
            "--w_pattern", args.w_pattern,
            "--w_mask_shape", args.w_mask_shape,
            "--w_up_radius", str(args.w_up_radius),
            "--w_low_radius", str(args.w_low_radius),
            "--prompts_file", str(prompts_path),
            "--dump_scores_json", str(dump_file.resolve()),
        ]

        if args.reference_model:
            cmd.extend(["--reference_model", args.reference_model])
        if args.reference_model_pretrain:
            cmd.extend(["--reference_model_pretrain", args.reference_model_pretrain])

        proc = subprocess.run(cmd, cwd=str(args.robin_root), check=False)
        if proc.returncode != 0:
            print(f"[WARN] DDIM step={n_steps} failed with exit code {proc.returncode}")
        else:
            print(f"[DDIM] step={n_steps} done → {dump_file}")


def _build_manifest_and_tasks_from_dump(
    score_dump_json: Path,
    data_root: Path,
    output_root: Path,
    combo_attacks: List[Tuple[str, List[str], int]] | None = None,
) -> Path:
    payload = _load_json(score_dump_json)
    dump = payload.get("score_dump", {})

    # Determine which attacks to process
    if combo_attacks is not None:
        attack_keys = [key for key, _parts, _sev in combo_attacks]
    else:
        # Legacy: 7 single attacks
        attack_keys = ATTACKS_7
        missing = [a for a in attack_keys if a not in dump]
        if missing:
            raise ValueError(f"score_dump missing required attacks: {missing}")

    tasks = []
    for key in attack_keys:
        if key not in dump:
            print(f"[WARN] attack '{key}' not in score_dump, skipping")
            continue
        task_id = f"robin_{key}"
        task_file = output_root / "tasks" / f"{task_id}.json"
        attack_entry = dump[key]
        attack_payload: Dict[str, Any] = {
            "score_dump": {
                "clean_scores": attack_entry["clean_scores"],
                "watermarked_scores": attack_entry["watermarked_scores"],
            }
        }
        # Include ring features when available (multi-dim detection)
        if "clean_ring_features" in attack_entry and "watermarked_ring_features" in attack_entry:
            attack_payload["score_dump"]["clean_ring_features"] = attack_entry["clean_ring_features"]
            attack_payload["score_dump"]["watermarked_ring_features"] = attack_entry["watermarked_ring_features"]

        # Include severity metadata for combo tasks
        if combo_attacks is not None:
            match = [c for c in combo_attacks if c[0] == key]
            if match:
                _k, parts, severity = match[0]
                attack_payload["severity"] = severity
                attack_payload["attack_components"] = parts

        _write_json(task_file, attack_payload)
        tasks.append({"task_id": task_id, "score_json": str(task_file.relative_to(data_root)).replace("\\", "/")})

    manifest = {
        "calibration_task_id": "robin_none",
        "tasks": tasks,
    }
    manifest_path = data_root / "task_manifest.json"
    _write_json(manifest_path, manifest)

    # Emit severity_map.json for downstream metrics
    if combo_attacks is not None:
        sev_map = {}
        for key, _parts, severity in combo_attacks:
            task_id = f"robin_{key}"
            sev_map[task_id] = severity
        _write_json(output_root / "severity_map.json", sev_map)

    return manifest_path


def _build_meta_config(
    args: argparse.Namespace,
    data_root: Path,
    output_root: Path,
    algo: str,
    train_ids: List[str] | None = None,
    val_ids: List[str] | None = None,
    test_ids: List[str] | None = None,
) -> Path:
    # Use provided split or fall back to legacy hardcoded split
    if train_ids is None:
        train_ids = ["robin_none", "robin_jpeg", "robin_blurring", "robin_color_jitter"]
    if val_ids is None:
        val_ids = ["robin_cropping"]
    if test_ids is None:
        test_ids = ["robin_rotation", "robin_noise"]

    cfg = {
        "data_root": str(data_root).replace("\\", "/"),
        "output_root": str(output_root / "meta_out").replace("\\", "/"),
        "global_seed": int(args.global_seed),
        "split_seed": int(args.split_seed),
        "meta_train_task_ids": train_ids,
        "meta_val_task_ids": val_ids,
        "meta_test_task_ids": test_ids,
        "support_size": int(args.support_size),
        "query_size": int(args.query_size),
        "alpha_fpr": float(args.alpha_fpr),
        "tpr_target_beta": float(args.tpr_target_beta),
        "k_grid": [int(v) for v in args.k_grid.split(",")],
        "wm_checkpoint_path": args.wm_path,
        "algorithm": algo,
        "inner_steps": int(args.inner_steps),
        "inner_lr": float(args.inner_lr),
        "meta_lr": float(args.meta_lr),
        "outer_iterations": int(args.outer_iterations),
        "meta_batch_size_tasks": int(args.meta_batch_size_tasks),
        "checkpoint_eval_every": int(args.checkpoint_eval_every),
        "device": args.device,
    }
    cfg_path = output_root / f"meta_config_{algo}.json"
    _write_json(cfg_path, cfg)
    return cfg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ROBIN runtime bridge to MWDRAS meta-learning")
    parser.add_argument("--robin-root", default="d:/RCE/robin_official")
    parser.add_argument("--bridge-output-root", default="d:/RCE/outputs_p4_runtime")

    parser.add_argument("--run-name", default="mwdras_runtime_bridge")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=16)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--wm-path", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--test-num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--reference-model", default=None)
    parser.add_argument("--reference-model-pretrain", default=None)
    parser.add_argument("--prompts-file", default=None)

    parser.add_argument("--w-channel", type=int, default=3)
    parser.add_argument("--w-pattern", default="ring")
    parser.add_argument("--w-mask-shape", default="circle")
    parser.add_argument("--w-up-radius", type=int, default=15)
    parser.add_argument("--w-low-radius", type=int, default=5)

    parser.add_argument("--algorithm", default="fomaml", choices=["fomaml", "reptile"])
    parser.add_argument("--global-seed", type=int, default=123)
    parser.add_argument("--split-seed", type=int, default=456)
    parser.add_argument("--support-size", type=int, default=8)
    parser.add_argument("--query-size", type=int, default=8)
    parser.add_argument("--alpha-fpr", type=float, default=0.05)
    parser.add_argument("--tpr-target-beta", type=float, default=0.9)
    parser.add_argument("--k-grid", default="0,1,2,4,8")
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--inner-lr", type=float, default=0.3)
    parser.add_argument("--meta-lr", type=float, default=0.1)
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--meta-batch-size-tasks", type=int, default=2)
    parser.add_argument("--checkpoint-eval-every", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--skip-robin-run", action="store_true", help="Use existing score dump file instead of running ROBIN")
    parser.add_argument("--existing-score-dump", default=None, help="Path to existing ROBIN score dump JSON")
    parser.add_argument("--combo-mode", action="store_true",
                        help="Use composite attack framework (2^6 task subsets) instead of 7 fixed attacks")
    parser.add_argument("--max-severity", type=int, default=6,
                        help="Maximum combo severity level (1-6). Default: 6 (all subsets)")
    parser.add_argument("--ddim-step-sweep", default=None, type=str,
                        help="Comma-separated DDIM test step counts for Q3 tradeoff (e.g. 5,10,25,50)")

    args = parser.parse_args()

    output_root = Path(args.bridge_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    score_dump_json = output_root / "robin_runtime_score_dump.json"

    if not args.prompts_file:
        default_prompts = output_root / "runtime_prompts.txt"
        if not default_prompts.exists():
            default_prompts.write_text("a high quality photo of a mountain lake at sunrise\n", encoding="utf-8")
        args.prompts_file = str(default_prompts)

    # Build combo attack list if combo mode
    combo_attacks = None
    if args.combo_mode:
        combo_attacks = _generate_combo_attacks(max_severity=args.max_severity)
        attack_keys = [key for key, _parts, _sev in combo_attacks]
        print(f"[COMBO] Generated {len(combo_attacks)} tasks across severity 0-{args.max_severity}")
    else:
        attack_keys = None

    if args.skip_robin_run:
        if not args.existing_score_dump:
            raise ValueError("--existing-score-dump is required with --skip-robin-run")
        score_dump_json = Path(args.existing_score_dump)
    else:
        _run_robin_inject(args, score_dump_json, attack_keys=attack_keys)

    data_root = output_root
    manifest_path = _build_manifest_and_tasks_from_dump(
        score_dump_json, data_root, output_root, combo_attacks=combo_attacks
    )
    n_tasks = len(combo_attacks) if combo_attacks else 7
    print(f"[OK] {n_tasks} attacks captured and manifest generated: {manifest_path}")

    # Determine train/val/test split
    if args.combo_mode and combo_attacks is not None:
        train_ids, val_ids, test_ids = _split_combo_tasks(combo_attacks, seed=args.split_seed)
        print(f"[SPLIT] train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    else:
        train_ids, val_ids, test_ids = None, None, None

    cfg_path = _build_meta_config(args, data_root, output_root, args.algorithm,
                                   train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
    print(f"[OK] meta config generated: {cfg_path}")

    # DDIM step sweep for Q3 (runs before meta runner so scores are available)
    if args.ddim_step_sweep and not args.skip_robin_run:
        step_counts = [int(s.strip()) for s in args.ddim_step_sweep.split(",")]
        _run_ddim_step_sweep(args, output_root, step_counts)

    print("[RUN] Starting MWDRAS meta runner from runtime ROBIN values...")
    run_meta(cfg_path)
    print("[DONE] bridge pipeline complete.")


if __name__ == "__main__":
    main()
