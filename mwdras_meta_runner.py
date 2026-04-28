from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from mwdras_meta_learners import (
    FOMAMLLearner,
    ReptileLearner,
    RecoveryConfig,
    TaskData,
    evaluate_recovery_kstar,
    quantile_threshold,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_distances(payload: dict, attack_key: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supported score_dump formats from existing evaluation scripts:
    1) {score_dump: {attack: {clean_scores, watermarked_scores}}}  -- scalar (legacy)
    2) {score_dump: {attack: {clean_ring_features, watermarked_ring_features}}}  -- per-ring vectors
    3) {score_dump: {clean_scores, watermarked_scores}}

    When ring features are available, returns (N, D) arrays; otherwise (N,).
    """
    score_dump = payload.get("score_dump")
    if score_dump is None:
        raise ValueError("Missing 'score_dump'. Re-run evaluation with --dump-scores.")

    if isinstance(score_dump, dict) and attack_key and attack_key in score_dump:
        entry = score_dump[attack_key]
        # Prefer ring features if present
        if "clean_ring_features" in entry and "watermarked_ring_features" in entry:
            return (
                np.asarray(entry["clean_ring_features"], dtype=np.float32),
                np.asarray(entry["watermarked_ring_features"], dtype=np.float32),
            )
        return np.asarray(entry["clean_scores"], dtype=float), np.asarray(entry["watermarked_scores"], dtype=float)

    if isinstance(score_dump, dict) and "clean_scores" in score_dump and "watermarked_scores" in score_dump:
        # Prefer ring features if present
        if "clean_ring_features" in score_dump and "watermarked_ring_features" in score_dump:
            return (
                np.asarray(score_dump["clean_ring_features"], dtype=np.float32),
                np.asarray(score_dump["watermarked_ring_features"], dtype=np.float32),
            )
        return np.asarray(score_dump["clean_scores"], dtype=float), np.asarray(score_dump["watermarked_scores"], dtype=float)

    raise ValueError("Unsupported score_dump format for current task.")


def _to_detection_feature(distance_scores: np.ndarray) -> np.ndarray:
    # ROBIN distance is lower for watermarked samples; negate so higher = more watermarked.
    # Works for both 1D (N,) and multi-dim (N, D) arrays.
    return -distance_scores


def _resolve_device(cfg: Dict[str, Any]) -> str:
    requested = str(cfg.get("device", "auto")).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_task(
    task_id: str,
    clean_dist: np.ndarray,
    wm_dist: np.ndarray,
    support_size: int,
    query_size: int,
    seed: int,
    device: str,
    feat_mean: np.ndarray | None = None,
    feat_std: np.ndarray | None = None,
) -> TaskData:
    rng = np.random.default_rng(seed)

    clean_feat = _to_detection_feature(clean_dist)
    wm_feat = _to_detection_feature(wm_dist)

    # Apply z-score normalization if stats are provided
    if feat_mean is not None and feat_std is not None:
        clean_feat = (clean_feat - feat_mean) / feat_std
        wm_feat = (wm_feat - feat_mean) / feat_std

    if len(clean_feat) < support_size + query_size or len(wm_feat) < support_size + query_size:
        raise ValueError(
            f"Task {task_id} does not have enough samples for support/query split: "
            f"need {support_size + query_size}, have clean={len(clean_feat)}, wm={len(wm_feat)}"
        )

    clean_idx = rng.permutation(len(clean_feat))
    wm_idx = rng.permutation(len(wm_feat))

    clean_support = clean_feat[clean_idx[:support_size]]
    clean_query = clean_feat[clean_idx[support_size:support_size + query_size]]
    wm_support = wm_feat[wm_idx[:support_size]]
    wm_query = wm_feat[wm_idx[support_size:support_size + query_size]]

    support_x = np.concatenate([clean_support, wm_support], axis=0)
    support_y = np.concatenate([np.zeros(len(clean_support)), np.ones(len(wm_support))], axis=0)
    query_x = np.concatenate([clean_query, wm_query], axis=0)
    query_y = np.concatenate([np.zeros(len(clean_query)), np.ones(len(wm_query))], axis=0)

    return TaskData(
        task_id=task_id,
        support_x=torch.tensor(support_x, dtype=torch.float32, device=device),
        support_y=torch.tensor(support_y, dtype=torch.float32, device=device),
        query_x=torch.tensor(query_x, dtype=torch.float32, device=device),
        query_y=torch.tensor(query_y, dtype=torch.float32, device=device),
    )


def _resolve_score_path(data_root: Path, score_json: str) -> Path:
    p = Path(score_json)
    return p if p.is_absolute() else data_root / p


class _FullRetrainBaseline:
    """Task-specific scratch training baseline used for meta-vs-retrain comparison."""

    def __init__(self, inner_lr: float, seed: int, feat_dim: int = 1, device: str = "cpu") -> None:
        self.inner_lr = float(inner_lr)
        self.rng = np.random.default_rng(seed)
        self.feat_dim = feat_dim
        self.device = device

    def adapt_k_steps(self, task: TaskData, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Retrain baseline starts from random init for each task.
        w0 = torch.tensor(self.rng.normal(0.0, 0.02, size=self.feat_dim).astype(np.float32), device=self.device)
        b0 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        w = w0.clone().detach().requires_grad_(True)
        b = b0.clone().detach().requires_grad_(True)

        for _ in range(int(k)):
            logits = task.support_x @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, task.support_y)
            gw, gb = torch.autograd.grad(loss, [w, b], create_graph=False)
            w = (w - self.inner_lr * gw).detach().requires_grad_(True)
            b = (b - self.inner_lr * gb).detach().requires_grad_(True)

        return w.detach(), b.detach()

    def predict_scores(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (x @ w + b).detach()


class _GenericFineTuneBaseline:
    """B2: Start from the mean of train-task adapted weights, then fine-tune on test support.

    This represents the standard 'pretrained checkpoint → fine-tune' paradigm.
    The generic checkpoint is computed once by adapting meta-init on each train task
    and averaging the resulting weights.
    """

    def __init__(self, generic_w: torch.Tensor, generic_b: torch.Tensor,
                 inner_lr: float, device: str = "cpu") -> None:
        self.generic_w = generic_w.to(device).detach()
        self.generic_b = generic_b.to(device).detach()
        self.inner_lr = float(inner_lr)
        self.device = device

    def adapt_k_steps(self, task: TaskData, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.generic_w.clone().detach().requires_grad_(True)
        b = self.generic_b.clone().detach().requires_grad_(True)

        for _ in range(int(k)):
            logits = task.support_x @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, task.support_y)
            gw, gb = torch.autograd.grad(loss, [w, b], create_graph=False)
            w = (w - self.inner_lr * gw).detach().requires_grad_(True)
            b = (b - self.inner_lr * gb).detach().requires_grad_(True)

        return w.detach(), b.detach()

    def predict_scores(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (x @ w + b).detach()


class _ThresholdOnlyBaseline:
    """B3: Keep meta-init weights frozen (k=0), re-calibrate threshold τ from test support.

    No weight adaptation at all — only the decision threshold is adjusted per-task
    using the support set clean scores.
    """

    def __init__(self, frozen_w: torch.Tensor, frozen_b: torch.Tensor,
                 alpha_fpr: float, device: str = "cpu") -> None:
        self.frozen_w = frozen_w.to(device).detach()
        self.frozen_b = frozen_b.to(device).detach()
        self.alpha_fpr = alpha_fpr
        self.device = device

    def adapt_k_steps(self, task: TaskData, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Weights are always frozen; k is ignored.
        return self.frozen_w.clone(), self.frozen_b.clone()

    def predict_scores(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (x @ w + b).detach()

    def recalibrate_tau(self, task: TaskData) -> float:
        """Compute per-task threshold from support clean scores."""
        clean_mask = task.support_y == 0
        if clean_mask.sum() == 0:
            return 0.0
        clean_scores = self.predict_scores(
            task.support_x[clean_mask], self.frozen_w, self.frozen_b
        ).cpu().numpy()
        return float(np.quantile(clean_scores, 1.0 - self.alpha_fpr))


def _task_metrics_at_k(learner: Any, task: TaskData, tau_alpha: float, k: int) -> Dict[str, float]:
    t0 = perf_counter()
    w, b = learner.adapt_k_steps(task, int(k))
    elapsed_ms = (perf_counter() - t0) * 1000.0

    clean = learner.predict_scores(task.query_x[task.query_y == 0], w, b).cpu().numpy()
    wm = learner.predict_scores(task.query_x[task.query_y == 1], w, b).cpu().numpy()

    tpr = float(np.mean(wm > tau_alpha))
    fpr = float(np.mean(clean > tau_alpha))

    y_true = np.concatenate([np.zeros_like(clean), np.ones_like(wm)])
    y_score = np.concatenate([clean, wm])
    fpr_curve, tpr_curve, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    auc = float(metrics.auc(fpr_curve, tpr_curve))

    return {
        "k": int(k),
        "tpr": tpr,
        "fpr": fpr,
        "auc": auc,
        "latency_ms": float(elapsed_ms),
    }


def _select_k_star(step_rows: List[Dict[str, float]], alpha_fpr: float, beta_tpr: float) -> int | None:
    for row in step_rows:
        if row["tpr"] >= beta_tpr and row["fpr"] <= alpha_fpr:
            return int(row["k"])
    return None


def _compare_meta_vs_retrain(
    meta_learner: Any,
    retrain_learner: Any,
    tasks: List[TaskData],
    tau_alpha: float,
    rec_cfg: RecoveryConfig,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    delta_k_vals: List[float] = []

    for task in tasks:
        meta_steps = [_task_metrics_at_k(meta_learner, task, tau_alpha, k) for k in rec_cfg.k_grid]
        retrain_steps = [_task_metrics_at_k(retrain_learner, task, tau_alpha, k) for k in rec_cfg.k_grid]

        k_meta = _select_k_star(meta_steps, rec_cfg.alpha_fpr, rec_cfg.tpr_target_beta)
        k_retrain = _select_k_star(retrain_steps, rec_cfg.alpha_fpr, rec_cfg.tpr_target_beta)
        delta_k = None
        if k_meta is not None and k_retrain is not None:
            delta_k = float(k_meta - k_retrain)
            delta_k_vals.append(delta_k)

        rows.append(
            {
                "task_id": task.task_id,
                "k_star_meta": k_meta,
                "k_star_retrain": k_retrain,
                "delta_k_meta_minus_retrain": delta_k,
            }
        )

    return {
        "task_rows": rows,
        "delta_k_mean": float(np.mean(delta_k_vals)) if delta_k_vals else None,
        "meta_faster_on_steps": bool(np.mean(delta_k_vals) < 0.0) if delta_k_vals else None,
    }


def _bound_diagnostics(meta_learner: Any, tasks: List[TaskData], tau_alpha: float, rec_cfg: RecoveryConfig) -> Dict[str, Any]:
    task_reports: List[Dict[str, Any]] = []
    rho_all: List[float] = []

    for task in tasks:
        errors: List[float] = []
        for k in rec_cfg.k_grid:
            m = _task_metrics_at_k(meta_learner, task, tau_alpha, k)
            # Recovery-error proxy aligned with fixed-FPR objective.
            e = max(0.0, rec_cfg.tpr_target_beta - m["tpr"]) + max(0.0, m["fpr"] - rec_cfg.alpha_fpr)
            errors.append(float(e))

        ratios: List[float] = []
        for i in range(len(errors) - 1):
            if errors[i] > 1e-9:
                ratios.append(errors[i + 1] / errors[i])
        rho_hat = float(np.median(ratios)) if ratios else 1.0
        rho_hat = min(max(rho_hat, 0.0), 1.0)
        rho_all.append(rho_hat)

        e0 = errors[0] if errors else 0.0
        residuals = [errors[i] - ((rho_hat ** i) * e0) for i in range(len(errors))]
        c_delta_hat = float(max(residuals)) if residuals else 0.0
        monotone_noninc = bool(all(errors[i + 1] <= errors[i] + 1e-9 for i in range(len(errors) - 1)))

        task_reports.append(
            {
                "task_id": task.task_id,
                "error_curve": errors,
                "rho_hat": rho_hat,
                "c_delta_hat": c_delta_hat,
                "monotone_nonincreasing": monotone_noninc,
            }
        )

    return {
        "tasks": task_reports,
        "rho_hat_mean": float(np.mean(rho_all)) if rho_all else None,
        "condition_0_lt_rho_lt_1": bool(0.0 < np.mean(rho_all) < 1.0) if rho_all else None,
    }


def _amortized_tradeoff(meta_learner: Any, tasks: List[TaskData], tau_alpha: float, rec_cfg: RecoveryConfig,
                        ddim_step_scores: Dict[int, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    delta_auc: List[float] = []
    delta_lat: List[float] = []

    for task in tasks:
        # Standard verifier: task-optimal step (k* if exists else max k)
        step_rows = [_task_metrics_at_k(meta_learner, task, tau_alpha, k) for k in rec_cfg.k_grid]
        k_star = _select_k_star(step_rows, rec_cfg.alpha_fpr, rec_cfg.tpr_target_beta)
        k_std = int(k_star) if k_star is not None else int(max(rec_cfg.k_grid))
        std_row = next(r for r in step_rows if int(r["k"]) == k_std)

        # Amortized verifier: zero-step inference (no per-task adaptation).
        am_row = next(r for r in step_rows if int(r["k"]) == 0)

        d_auc = float(std_row["auc"] - am_row["auc"])
        d_lat = float(std_row["latency_ms"] - am_row["latency_ms"])
        delta_auc.append(d_auc)
        delta_lat.append(d_lat)

        rows.append(
            {
                "task_id": task.task_id,
                "k_standard": k_std,
                "auc_standard": float(std_row["auc"]),
                "latency_standard_ms": float(std_row["latency_ms"]),
                "auc_amortized": float(am_row["auc"]),
                "latency_amortized_ms": float(am_row["latency_ms"]),
                "delta_auc_standard_minus_amortized": d_auc,
                "delta_latency_standard_minus_amortized_ms": d_lat,
            }
        )

    result: Dict[str, Any] = {
        "task_rows": rows,
        "delta_auc_mean": float(np.mean(delta_auc)) if delta_auc else None,
        "delta_latency_mean_ms": float(np.mean(delta_lat)) if delta_lat else None,
        "tradeoff_observed": bool((np.mean(delta_auc) > 0.0) and (np.mean(delta_lat) > 0.0)) if delta_auc and delta_lat else None,
    }

    # DDIM step tradeoff (C3): if score dumps at multiple inference steps are available
    if ddim_step_scores:
        result["ddim_step_tradeoff"] = _ddim_step_tradeoff(ddim_step_scores)

    return result


def _ddim_step_tradeoff(ddim_step_scores: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute AUC at each DDIM inference step count.

    ddim_step_scores: {num_steps: {attack_key: {clean_scores, watermarked_scores}}}
    """
    pareto_points: List[Dict[str, Any]] = []

    for num_steps in sorted(ddim_step_scores.keys()):
        attack_data = ddim_step_scores[num_steps]
        auc_vals: List[float] = []
        for attack_key, entry in attack_data.items():
            clean = np.asarray(entry["clean_scores"], dtype=float)
            wm = np.asarray(entry["watermarked_scores"], dtype=float)
            y_true = np.concatenate([np.zeros_like(clean), np.ones_like(wm)])
            y_score = np.concatenate([-clean, -wm])  # negate: lower distance = watermarked
            fpr_c, tpr_c, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
            auc_vals.append(float(metrics.auc(fpr_c, tpr_c)))

        pareto_points.append({
            "num_inference_steps": int(num_steps),
            "mean_auc": float(np.mean(auc_vals)),
            "per_attack_auc": {k: float(v) for k, v in zip(attack_data.keys(), auc_vals)},
            "n_attacks": len(auc_vals),
        })

    return {
        "pareto_curve": pareto_points,
        "step_counts": sorted(ddim_step_scores.keys()),
    }


def _compare_all_baselines(
    meta_learner: Any,
    retrain_learner: Any,
    b2_learner: Any,
    b3_learner: _ThresholdOnlyBaseline,
    tasks: List[TaskData],
    tau_alpha: float,
    rec_cfg: RecoveryConfig,
) -> Dict[str, Any]:
    """Compare meta vs B1 (retrain), B2 (generic fine-tune), B3 (threshold-only)."""
    rows: List[Dict[str, Any]] = []

    for task in tasks:
        row: Dict[str, Any] = {"task_id": task.task_id}

        # Meta
        meta_steps = [_task_metrics_at_k(meta_learner, task, tau_alpha, k) for k in rec_cfg.k_grid]
        row["k_star_meta"] = _select_k_star(meta_steps, rec_cfg.alpha_fpr, rec_cfg.tpr_target_beta)
        row["auc_meta_max_k"] = meta_steps[-1]["auc"] if meta_steps else None

        # B1: full retrain
        b1_steps = [_task_metrics_at_k(retrain_learner, task, tau_alpha, k) for k in rec_cfg.k_grid]
        row["k_star_b1_retrain"] = _select_k_star(b1_steps, rec_cfg.alpha_fpr, rec_cfg.tpr_target_beta)

        # B2: generic fine-tune
        b2_steps = [_task_metrics_at_k(b2_learner, task, tau_alpha, k) for k in rec_cfg.k_grid]
        row["k_star_b2_generic_ft"] = _select_k_star(b2_steps, rec_cfg.alpha_fpr, rec_cfg.tpr_target_beta)

        # B3: threshold-only (no adaptation, re-calibrated tau)
        tau_b3 = b3_learner.recalibrate_tau(task)
        b3_steps = [_task_metrics_at_k(b3_learner, task, tau_b3, k) for k in [0]]
        row["k_star_b3_threshold"] = 0 if (b3_steps[0]["tpr"] >= rec_cfg.tpr_target_beta and b3_steps[0]["fpr"] <= rec_cfg.alpha_fpr) else None
        row["tpr_b3"] = b3_steps[0]["tpr"]
        row["fpr_b3"] = b3_steps[0]["fpr"]
        row["auc_b3"] = b3_steps[0]["auc"]
        row["tau_b3_recalibrated"] = float(tau_b3)

        rows.append(row)

    # Summary
    methods = {
        "meta": [r["k_star_meta"] for r in rows],
        "b1_retrain": [r["k_star_b1_retrain"] for r in rows],
        "b2_generic_ft": [r["k_star_b2_generic_ft"] for r in rows],
        "b3_threshold": [r["k_star_b3_threshold"] for r in rows],
    }
    summary: Dict[str, Any] = {}
    for name, ks in methods.items():
        valid = [k for k in ks if k is not None]
        summary[name] = {
            "recovered": len(valid),
            "total": len(ks),
            "mean_k_star": float(np.mean(valid)) if valid else None,
        }

    return {"task_rows": rows, "summary": summary}


def _val_objective_from_results(task_results: List[Dict[str, Any]]) -> Tuple[int, float, float, float]:
    """
    Paper-oriented selection objective (lower is better):
    1) number of unrecovered tasks
    2) mean k* over recovered tasks
    3) mean time_to_target over recovered tasks
    4) mean adaptation latency over recovered tasks
    """
    unrecovered = 0
    k_vals: List[float] = []
    t_vals: List[float] = []
    lat_vals: List[float] = []

    for r in task_results:
        if r.get("k_star") is None or r.get("time_to_target_sec") is None:
            unrecovered += 1
            continue
        k_vals.append(float(r["k_star"]))
        t_vals.append(float(r["time_to_target_sec"]))
        step_metrics = r.get("step_metrics", [])
        if step_metrics:
            lat_vals.extend(float(s.get("latency_ms", 0.0)) for s in step_metrics)

    mean_k = float(np.mean(k_vals)) if k_vals else float("inf")
    mean_t = float(np.mean(t_vals)) if t_vals else float("inf")
    mean_lat = float(np.mean(lat_vals)) if lat_vals else float("inf")
    return (unrecovered, mean_k, mean_t, mean_lat)


def run(config_path: Path) -> None:
    cfg = _load_json(config_path)

    required = [
        "data_root",
        "output_root",
        "global_seed",
        "split_seed",
        "meta_train_task_ids",
        "meta_val_task_ids",
        "meta_test_task_ids",
        "support_size",
        "query_size",
        "alpha_fpr",
        "tpr_target_beta",
        "k_grid",
        "wm_checkpoint_path",
        "algorithm",
        "inner_steps",
        "inner_lr",
        "meta_lr",
        "outer_iterations",
        "meta_batch_size_tasks",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    data_root = Path(cfg["data_root"])
    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    # We intentionally do not load/modify ROBIN checkpoint; path is tracked for protocol reproducibility.
    _ = cfg["wm_checkpoint_path"]

    manifest_path = data_root / "task_manifest.json"
    manifest = _load_json(manifest_path)
    task_entries = {t["task_id"]: t for t in manifest["tasks"]}

    support_size = int(cfg["support_size"])
    query_size = int(cfg["query_size"])
    split_seed = int(cfg["split_seed"])
    device = _resolve_device(cfg)

    meta_train_ids: List[str] = list(cfg["meta_train_task_ids"])
    meta_val_ids: List[str] = list(cfg["meta_val_task_ids"])
    meta_test_ids: List[str] = list(cfg["meta_test_task_ids"])

    # Compute z-score normalization stats from training task features
    all_train_feats = []
    for tid in meta_train_ids:
        t = task_entries[tid]
        payload = _load_json(_resolve_score_path(data_root, t["score_json"]))
        clean_d, wm_d = _extract_distances(payload, t.get("attack_key"))
        all_train_feats.append(_to_detection_feature(clean_d))
        all_train_feats.append(_to_detection_feature(wm_d))
    all_train_feats_arr = np.concatenate(all_train_feats, axis=0)
    feat_mean = np.mean(all_train_feats_arr, axis=0)
    feat_std = np.std(all_train_feats_arr, axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)  # avoid division by zero
    print(f"[META] Feature normalization: mean={feat_mean}, std={feat_std}")

    def build_task(task_id: str, seed_offset: int) -> TaskData:
        t = task_entries[task_id]
        payload = _load_json(_resolve_score_path(data_root, t["score_json"]))
        clean_d, wm_d = _extract_distances(payload, t.get("attack_key"))
        return _make_task(task_id, clean_d, wm_d, support_size, query_size, split_seed + seed_offset, device,
                          feat_mean=feat_mean, feat_std=feat_std)

    train_tasks = [build_task(tid, i) for i, tid in enumerate(meta_train_ids)]
    val_tasks = [build_task(tid, 500 + i) for i, tid in enumerate(meta_val_ids)]
    test_tasks = [build_task(tid, 1000 + i) for i, tid in enumerate(meta_test_ids)]

    # Detect feature dimension from first train task
    feat_dim = train_tasks[0].support_x.shape[-1] if train_tasks[0].support_x.ndim > 1 else 1
    print(f"[META] Feature dimension: {feat_dim}")

    algo = str(cfg["algorithm"]).lower()
    if algo == "fomaml":
        learner = FOMAMLLearner(
            inner_lr=float(cfg["inner_lr"]),
            meta_lr=float(cfg["meta_lr"]),
            inner_steps=int(cfg["inner_steps"]),
            outer_iterations=int(cfg["outer_iterations"]),
            meta_batch_size_tasks=int(cfg["meta_batch_size_tasks"]),
            seed=int(cfg["global_seed"]),
            feat_dim=feat_dim,
            device=device,
        )
    elif algo == "reptile":
        learner = ReptileLearner(
            inner_lr=float(cfg["inner_lr"]),
            meta_lr=float(cfg["meta_lr"]),
            inner_steps=int(cfg["inner_steps"]),
            outer_iterations=int(cfg["outer_iterations"]),
            meta_batch_size_tasks=int(cfg["meta_batch_size_tasks"]),
            seed=int(cfg["global_seed"]),
            feat_dim=feat_dim,
            device=device,
        )
    else:
        raise ValueError("algorithm must be 'fomaml' or 'reptile'")

    calib_task_id = manifest.get("calibration_task_id")
    if calib_task_id is None:
        raise ValueError("task_manifest.json must include calibration_task_id")

    calib_task = build_task(calib_task_id, 5000)

    rec_cfg = RecoveryConfig(
        alpha_fpr=float(cfg["alpha_fpr"]),
        tpr_target_beta=float(cfg["tpr_target_beta"]),
        k_grid=[int(k) for k in cfg["k_grid"]],
    )

    # Train with periodic validation and keep best checkpoint by paper criteria.
    total_iters = int(cfg["outer_iterations"])
    eval_every = int(cfg.get("checkpoint_eval_every", max(1, total_iters // 10)))
    if eval_every <= 0:
        raise ValueError("checkpoint_eval_every must be >= 1")

    best_obj: Tuple[int, float, float, float] | None = None
    best_state: Dict[str, Any] | None = None
    trained_iters = 0

    while trained_iters < total_iters:
        step_iters = min(eval_every, total_iters - trained_iters)
        learner.outer_iterations = step_iters
        learner.meta_fit(train_tasks)
        trained_iters += step_iters

        # Recompute fixed-FPR threshold at current checkpoint using calibration task.
        w_cal, b_cal = learner.adapt_k_steps(calib_task, 0)
        clean_scores_cal = learner.predict_scores(calib_task.query_x[calib_task.query_y == 0], w_cal, b_cal).cpu().numpy()
        tau_cur = quantile_threshold(clean_scores_cal, float(cfg["alpha_fpr"]))

        val_results: List[Dict[str, Any]] = []
        for task in val_tasks:
            val_results.append(evaluate_recovery_kstar(learner, task, tau_cur, rec_cfg))
        obj = _val_objective_from_results(val_results)

        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_state = {
                "w": learner.head.w.detach().cpu(),
                "b": learner.head.b.detach().cpu(),
                "tau_alpha": float(tau_cur),
                "trained_outer_iterations": int(trained_iters),
                "val_objective": {
                    "unrecovered_tasks": int(obj[0]),
                    "mean_k_star": float(obj[1]),
                    "mean_time_to_target_sec": float(obj[2]),
                    "mean_latency_ms": float(obj[3]),
                },
            }

    if best_state is None:
        raise RuntimeError("Best-checkpoint selection failed: no validation state was captured.")

    # Save best checkpoint requested for paper protocol.
    best_ckpt_file = output_root / f"meta_{algo}_best.pt"
    torch.save(best_state, best_ckpt_file)

    # Use best checkpoint for final test evaluation.
    learner.head.w = best_state["w"].to(device)
    learner.head.b = best_state["b"].to(device)
    tau_alpha = float(best_state["tau_alpha"])

    # Additional paper checks: Q1 (meta vs retrain), Q2 (bound diagnostics), Q3 (amortized tradeoff).
    retrain_baseline = _FullRetrainBaseline(inner_lr=float(cfg["inner_lr"]), seed=int(cfg["global_seed"]) + 17, feat_dim=feat_dim, device=device)

    # B2: generic checkpoint = mean of train-task adapted weights
    adapted_ws: List[torch.Tensor] = []
    adapted_bs: List[torch.Tensor] = []
    for task in train_tasks:
        w_a, b_a = learner.adapt_k_steps(task, int(cfg["inner_steps"]))
        adapted_ws.append(w_a)
        adapted_bs.append(b_a)
    generic_w = torch.stack(adapted_ws).mean(dim=0)
    generic_b = torch.stack(adapted_bs).mean(dim=0)
    b2_baseline = _GenericFineTuneBaseline(generic_w, generic_b, inner_lr=float(cfg["inner_lr"]), device=device)

    # B3: threshold-only (frozen meta-init weights, re-calibrate τ per task)
    b3_baseline = _ThresholdOnlyBaseline(learner.head.w, learner.head.b,
                                         alpha_fpr=float(cfg["alpha_fpr"]), device=device)

    q1_comparison = _compare_meta_vs_retrain(learner, retrain_baseline, test_tasks, tau_alpha, rec_cfg)
    q1_all_baselines = _compare_all_baselines(learner, retrain_baseline, b2_baseline, b3_baseline,
                                               test_tasks, tau_alpha, rec_cfg)
    q2_bound_val = _bound_diagnostics(learner, val_tasks, tau_alpha, rec_cfg)
    q2_bound_test = _bound_diagnostics(learner, test_tasks, tau_alpha, rec_cfg)

    # Q3: DDIM step tradeoff — load multi-step score dumps if available
    ddim_step_scores: Dict[int, Dict[str, Any]] | None = None
    ddim_steps_dir = data_root / "ddim_steps"
    if ddim_steps_dir.is_dir():
        ddim_step_scores = {}
        for step_file in sorted(ddim_steps_dir.glob("scores_step_*.json")):
            step_num = int(step_file.stem.split("_")[-1])
            step_payload = _load_json(step_file)
            step_dump = step_payload.get("score_dump", {})
            ddim_step_scores[step_num] = step_dump
        if not ddim_step_scores:
            ddim_step_scores = None
        else:
            print(f"[Q3] Loaded DDIM step scores for steps: {sorted(ddim_step_scores.keys())}")

    q3_tradeoff = _amortized_tradeoff(learner, test_tasks, tau_alpha, rec_cfg,
                                       ddim_step_scores=ddim_step_scores)

    results = {
        "algorithm": algo,
        "tau_alpha": tau_alpha,
        "alpha_fpr": rec_cfg.alpha_fpr,
        "tpr_target_beta": rec_cfg.tpr_target_beta,
        "selection": {
            "criterion": "min(unrecovered, mean_k_star, mean_time_to_target, mean_latency)",
            "best_checkpoint": str(best_ckpt_file),
            "trained_outer_iterations": int(best_state["trained_outer_iterations"]),
            "val_objective": best_state["val_objective"],
        },
        "paper_questions": {
            "q1_meta_vs_retrain": q1_comparison,
            "q1_all_baselines": q1_all_baselines,
            "q2_adaptation_bound_val": q2_bound_val,
            "q2_adaptation_bound_test": q2_bound_test,
            "q3_amortized_verifier_tradeoff": q3_tradeoff,
        },
        "tasks": [],
    }

    for task in test_tasks:
        results["tasks"].append(evaluate_recovery_kstar(learner, task, tau_alpha, rec_cfg))

    kstars = [t["k_star"] for t in results["tasks"] if t["k_star"] is not None]
    times = [t["time_to_target_sec"] for t in results["tasks"] if t["time_to_target_sec"] is not None]
    results["k_star_mean"] = float(np.mean(kstars)) if kstars else None
    results["time_to_target_mean_sec"] = float(np.mean(times)) if times else None

    out_file = output_root / f"meta_{algo}_results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Keep final checkpoint export for compatibility.
    ckpt_file = output_root / f"meta_{algo}_init.pt"
    torch.save({"w": learner.head.w.cpu(), "b": learner.head.b.cpu(), "tau_alpha": tau_alpha}, ckpt_file)

    print(f"[DONE] results: {out_file}")
    print(f"[DONE] best checkpoint: {best_ckpt_file}")
    print(f"[DONE] checkpoint: {ckpt_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MWDRAS meta-learning runner (FOMAML/Reptile)")
    parser.add_argument("--config", required=True, help="Path to minimal MWDRAS JSON config")
    args = parser.parse_args()
    run(Path(args.config))
