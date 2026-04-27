# MWDRAS

**Meta-Watermarking for Detection Recovery under Attack Shift**

> Official implementation of the paper:  
> *"Meta-Watermarking for Detection Recovery under Attack Shift: Empirical Analysis of FOMAML-Based Fast Adaptation"*  
> IEEE Access, 2026.

[![GitHub Pages](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-blue?logo=github)](https://Michael-John-73.github.io/mwdras/)
[![IEEE Access](https://img.shields.io/badge/Paper-IEEE%20Access-orange)](https://ieeexplore.ieee.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green?logo=python)](requirements.txt)

---

## Pipeline

![Pipeline Overview](docs/assets/fig1_pipeline_flow.png)

---

## Overview

MWDRAS addresses a practical weakness of ring-pattern watermarks in diffusion models: detection accuracy degrades sharply when the image undergoes an unseen attack type at inference time. Rather than retraining a detector from scratch for every new attack, MWDRAS meta-trains a lightweight logistic detection head using **FOMAML** (First-Order Model-Agnostic Meta-Learning), so the head can recover to ≥90% TPR at ≤5% FPR within **1–4 gradient steps** on a small support set.

The watermark substrate is **ROBIN** ([Liu et al., NeurIPS 2024](https://arxiv.org/abs/2410.04votre)), used as-is without modification. MWDRAS operates entirely on the ROBIN score layer.

---

## Repository Structure

```
MWDRAS/
├── mwdras_bridge.py          # Entry point — connects ROBIN output to meta-learner
├── mwdras_meta_learners.py   # FOMAMLLearner, ReptileLearner, evaluation utilities
├── mwdras_meta_runner.py     # Meta-training + baseline evaluation pipeline
├── mwdras_result_metrics.py  # Severity metrics, MCS, SI, bootstrap CI
├── gen_figures.py            # Reproduces all paper figures (Fig. 2–6)
├── gen_flow.py               # Reproduces Fig. 1 pipeline diagram
├── robin_config.json         # ROBIN parameter mapping for bridge (edit before use)
└── requirements.txt
```

> **ROBIN source** is **not included**. Clone [robin_official](https://github.com/XuandongZhao/ROBIN) separately and set `--robin-root` accordingly.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Edit `robin_config.json`

Set your paths:
```json
"robin_root": { "value": "/path/to/robin_official" },
"model_id":   { "value": "stabilityai/stable-diffusion-2-1-base" },
"wm_path":    { "value": "/path/to/robin_official/ckpts/no_training_r5_15.pt" }
```

### 3. Run the full pipeline

```bash
python mwdras_bridge.py \
  --robin-root  /path/to/robin_official \
  --model-id    stabilityai/stable-diffusion-2-1-base \
  --wm-path     /path/to/ckpts/no_training_r5_15.pt \
  --end 64 \
  --bridge-output-root ./outputs
```

The pipeline:
1. Calls ROBIN (`inject_wm_inner_latent_robin.py`) to generate clean/watermarked score dumps across 7 attacks.
2. Builds the task manifest and per-attack feature files.
3. Runs FOMAML meta-training and evaluates Meta vs. Baselines B1–B3.
4. Saves results to `./outputs/meta_out/meta_fomaml_results.json`.

### 4. Skip ROBIN (use existing score dump)

```bash
python mwdras_bridge.py \
  --skip-robin-run \
  --existing-score-dump ./outputs/robin_runtime_score_dump.json \
  --model-id stabilityai/stable-diffusion-2-1-base \
  --wm-path  /path/to/ckpts/no_training_r5_15.pt \
  --bridge-output-root ./outputs
```

### 5. Reproduce figures

```bash
python gen_figures.py   # Fig. 2–6
python gen_flow.py      # Fig. 1 (pipeline diagram)
```

---

## Method

```
ROBIN score dump  ──►  Task split (train/val/test)
                              │
                    FOMAML meta-training (outer loop)
                              │
                    k* search (inner adaptation steps)
                              │
                    Evaluation: TPR @ k*, FPR, AUC
```

### Baselines

| ID | Name | Description |
|----|------|-------------|
| Meta | **FOMAML** | Meta-initialized head, k-step fast adaptation |
| B1 | Full Retrain | Random init, task-specific training |
| B2 | Generic Fine-tune | Mean of train-task adapted weights → fine-tune |
| B3 | Threshold-only | Frozen weights, per-task threshold recalibration |

### Key Results (scale = 64 images)

| Method | k* (mean) | TPR@k* | FPR@k* | Recovery Rate | Rel. Compute |
|--------|-----------|--------|--------|---------------|--------------|
| **Meta (FOMAML)** | **1** | **0.91** | 0.04 | **100%** | **1×** |
| B1 Full Retrain | 4 | 0.90 | 0.05 | 100% | 43,000× |
| B2 Generic FT | 2 | 0.88 | 0.06 | 86% | 2× |
| B3 Threshold | — | 0.72 | 0.05 | 0% | <1× |

<p align="center">
  <img src="docs/assets/fig2_recovery_kstar.png" width="48%" alt="Fig 2: k* by attack"/>
  <img src="docs/assets/fig3_tpr_vs_scale.png"   width="48%" alt="Fig 3: TPR vs scale"/>
</p>
<p align="center">
  <em>Left: k* by attack type (FOMAML reaches target at k*=1 for all attacks). &nbsp; Right: TPR across scales 32–256.</em>
</p>

<p align="center">
  <img src="docs/assets/fig5_cost_benefit.png"          width="48%" alt="Fig 5: Cost-Benefit"/>
  <img src="docs/assets/fig4_adaptation_trajectory.png" width="48%" alt="Fig 4: Adaptation"/>
</p>
<p align="center">
  <em>Left: Cost-benefit sweet spot ★ at scale=64. &nbsp; Right: TPR adaptation trajectory (converges within 1–2 steps).</em>
</p>

<p align="center">
  <img src="docs/assets/fig6_spearman_scatter.png" width="55%" alt="Fig 6: Spearman"/>
</p>
<p align="center">
  <em>Spearman ρ=0.87 (p&lt;0.01) — k* increases monotonically with attack severity (MCS=1.0).</em>
</p>

---

## ROBIN Integration

MWDRAS does **not** modify any ROBIN source files. The bridge file `mwdras_bridge.py` invokes ROBIN as a subprocess:

```
mwdras_bridge.py
    │
    ├─ subprocess ──► robin_official/inject_wm_inner_latent_robin.py
    │                  (generates score dump: clean_scores, watermarked_scores per attack)
    │
    └─ direct call ──► mwdras_meta_runner.run(cfg_path)
                        (reads score dump, builds tasks, runs FOMAML)
```

The only ROBIN variables exposed to MWDRAS are listed in `robin_config.json`:
- Watermark geometry: `w_channel`, `w_pattern`, `w_mask_shape`, `w_up_radius`, `w_low_radius`
- Diffusion settings: `model_id`, `wm_path`, `num_inference_steps`, `guidance_scale`

---

## Reproducibility

| Item | Value |
|------|-------|
| Watermark checkpoint | `ckpts/no_training_r5_15.pt` (random init, no training) |
| `global_seed` | 123 |
| `split_seed` | 456 |
| Calibration | Fixed threshold from no-attack support set (B3 recalibrates per task) |
| ROBIN substrate | Unmodified `robin_official/` |

---

## Citation

```bibtex
@article{yun2026mwdras,
  title   = {Meta-Watermarking for Detection Recovery under Attack Shift:
             Empirical Analysis of FOMAML-Based Fast Adaptation},
  author  = {Yun, Hyun-Gyo and Kim, Jae-Young and Lee, Hye-Jin},
  journal = {IEEE Access},
  year    = {2026}
}
```

---

## License

Source code: MIT License.  
ROBIN substrate ([robin_official](https://github.com/XuandongZhao/ROBIN)): refer to the original repository for licensing terms.
