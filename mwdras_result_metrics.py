from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mean(values: List[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def _extract_task_rows(result_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    algo = result_payload.get("algorithm")
    for task in result_payload.get("tasks", []):
        step_metrics = task.get("step_metrics", [])
        final_step = step_metrics[-1] if step_metrics else {}
        rows.append(
            {
                "algorithm": algo,
                "task_id": task.get("task_id"),
                "k_star": task.get("k_star"),
                "time_to_target_sec": task.get("time_to_target_sec"),
                "final_tpr": final_step.get("tpr"),
                "final_fpr": final_step.get("fpr"),
                "final_latency_ms": final_step.get("latency_ms"),
            }
        )
    return rows


def _compute_summary(rows: List[Dict[str, Any]], baseline_full_retrain_time_sec: Optional[float]) -> Dict[str, Any]:
    k_star_vals = [float(r["k_star"]) for r in rows if r.get("k_star") is not None]
    ttt_vals = [float(r["time_to_target_sec"]) for r in rows if r.get("time_to_target_sec") is not None]
    latency_vals = [float(r["final_latency_ms"]) for r in rows if r.get("final_latency_ms") is not None]

    recovered = len(k_star_vals)
    total = len(rows)

    summary: Dict[str, Any] = {
        "num_tasks": total,
        "num_recovered_tasks": recovered,
        "recovery_rate": (float(recovered) / float(total)) if total > 0 else None,
        "k_star_mean": _safe_mean(k_star_vals),
        "time_to_target_mean_sec": _safe_mean(ttt_vals),
        "final_latency_mean_ms": _safe_mean(latency_vals),
    }

    if baseline_full_retrain_time_sec is not None and summary["time_to_target_mean_sec"] is not None:
        if summary["time_to_target_mean_sec"] > 0:
            summary["speedup_vs_retrain"] = float(baseline_full_retrain_time_sec) / float(summary["time_to_target_mean_sec"])
        else:
            summary["speedup_vs_retrain"] = None

    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "algorithm",
        "task_id",
        "k_star",
        "time_to_target_sec",
        "final_tpr",
        "final_fpr",
        "final_latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ── Severity-aware metrics ──────────────────────────────────────────────────

def _group_by_severity(
    task_rows: List[Dict[str, Any]],
    severity_map: Dict[str, int],
) -> Dict[int, List[Dict[str, Any]]]:
    """Group task rows by severity level using an external mapping."""
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for row in task_rows:
        tid = row.get("task_id", "")
        sev = severity_map.get(tid)
        if sev is not None:
            groups.setdefault(sev, []).append(row)
    return groups


def _severity_metric_curve(
    groups: Dict[int, List[Dict[str, Any]]],
    metric_key: str,
) -> List[Tuple[int, float]]:
    """Compute mean of `metric_key` per severity level, sorted ascending."""
    curve = []
    for sev in sorted(groups.keys()):
        vals = [float(r[metric_key]) for r in groups[sev] if r.get(metric_key) is not None]
        if vals:
            curve.append((sev, float(np.mean(vals))))
    return curve


def compute_mcs(curve: List[Tuple[int, float]], delta: float = 0.0, higher_is_worse: bool = False) -> float:
    r"""Monotone Consistency Score.

    For metrics where higher_is_worse=True (e.g. k_star), checks non-decreasing:
    $$\mathbf{1}\!\bigl(\hat{\mu}(s_{\ell+1}) - \hat{\mu}(s_\ell) \ge -\delta\bigr)$$

    For metrics where higher_is_worse=False (e.g. AUC, TPR), checks non-increasing:
    $$\mathbf{1}\!\bigl(\hat{\mu}(s_\ell) - \hat{\mu}(s_{\ell+1}) \ge -\delta\bigr)$$
    """
    if len(curve) < 2:
        return 1.0
    count = 0
    for i in range(len(curve) - 1):
        diff = curve[i + 1][1] - curve[i][1]
        if higher_is_worse:
            # Expect non-decreasing (metric gets worse = higher)
            if diff >= -delta:
                count += 1
        else:
            # Expect non-increasing (metric gets worse = lower)
            if -diff >= -delta:  # i.e. diff <= delta
                count += 1
    return float(count) / float(len(curve) - 1)


def compute_si(curve: List[Tuple[int, float]]) -> Optional[float]:
    r"""Severity Index — mean slope of the metric curve.

    $$\mathrm{SI} = \frac{1}{L-1}\sum_{\ell=0}^{L-2}
    \frac{\hat{\mu}(s_{\ell+1}) - \hat{\mu}(s_\ell)}{s_{\ell+1} - s_\ell}$$

    Negative SI means the metric drops with severity (expected for AUC/TPR).
    """
    if len(curve) < 2:
        return None
    slopes = []
    for i in range(len(curve) - 1):
        ds = curve[i + 1][0] - curve[i][0]
        if ds == 0:
            continue
        slopes.append((curve[i + 1][1] - curve[i][1]) / float(ds))
    return float(np.mean(slopes)) if slopes else None


def bootstrap_ci(
    values: List[float],
    stat_fn=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic.

    Returns (point_estimate, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    point = float(stat_fn(arr))
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(float(stat_fn(sample)))
    boot_stats.sort()
    lo = boot_stats[int(n_boot * alpha / 2)]
    hi = boot_stats[int(n_boot * (1 - alpha / 2))]
    return point, lo, hi


def compute_severity_report(
    task_rows: List[Dict[str, Any]],
    severity_map: Dict[str, int],
    metric_key: str = "final_tpr",
    delta: float = 0.0,
) -> Dict[str, Any]:
    """Full severity analysis: curve, MCS, SI with bootstrap CIs."""
    groups = _group_by_severity(task_rows, severity_map)
    curve = _severity_metric_curve(groups, metric_key)

    # k_star increases with severity (higher = worse); AUC/TPR decrease (lower = worse)
    higher_is_worse = metric_key in ("k_star",)
    mcs = compute_mcs(curve, delta=delta, higher_is_worse=higher_is_worse)
    si = compute_si(curve)

    # Bootstrap CI for per-severity means
    severity_details = []
    for sev in sorted(groups.keys()):
        vals = [float(r[metric_key]) for r in groups[sev] if r.get(metric_key) is not None]
        if vals:
            pt, lo, hi = bootstrap_ci(vals)
            severity_details.append({
                "severity": sev,
                "n_tasks": len(vals),
                "mean": pt,
                "ci_95_low": lo,
                "ci_95_high": hi,
            })

    return {
        "metric": metric_key,
        "curve": [{"severity": s, "value": v} for s, v in curve],
        "mcs": mcs,
        "mcs_delta": delta,
        "si": si,
        "severity_details": severity_details,
    }


def run(result_json: Path, output_dir: Path, baseline_full_retrain_time_sec: Optional[float],
        severity_map: Optional[Dict[str, int]] = None) -> None:
    payload = _load_json(result_json)
    rows = _extract_task_rows(payload)
    summary = _compute_summary(rows, baseline_full_retrain_time_sec)

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = result_json.stem
    out_csv = output_dir / f"{stem}_task_metrics.csv"
    out_summary = output_dir / f"{stem}_summary.json"

    _write_csv(out_csv, rows)

    # Severity analysis if severity_map provided or can be inferred from task files
    if severity_map:
        for metric_key in ["final_tpr", "k_star"]:
            sev_report = compute_severity_report(rows, severity_map, metric_key=metric_key)
            summary[f"severity_{metric_key}"] = sev_report

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] task metrics: {out_csv}")
    print(f"[DONE] summary: {out_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MWDRAS result metrics calculator")
    parser.add_argument("--result-json", required=True, help="Output JSON from mwdras_meta_runner.py")
    parser.add_argument("--output-dir", default="outputs_p4/analysis", help="Directory for metrics CSV/JSON")
    parser.add_argument(
        "--baseline-full-retrain-time-sec",
        type=float,
        default=None,
        help="Optional baseline full retrain time (seconds) to compute speedup",
    )
    parser.add_argument(
        "--severity-map-json",
        default=None,
        help="JSON file mapping task_id -> severity (int). Auto-generated by combo bridge.",
    )
    args = parser.parse_args()

    severity_map = None
    if args.severity_map_json:
        with open(args.severity_map_json, "r", encoding="utf-8") as f:
            severity_map = {k: int(v) for k, v in json.load(f).items()}

    run(
        result_json=Path(args.result_json),
        output_dir=Path(args.output_dir),
        baseline_full_retrain_time_sec=args.baseline_full_retrain_time_sec,
        severity_map=severity_map,
    )


if __name__ == "__main__":
    main()
