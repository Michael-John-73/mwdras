from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Protocol, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TaskData:
	task_id: str
	support_x: torch.Tensor
	support_y: torch.Tensor
	query_x: torch.Tensor
	query_y: torch.Tensor


@dataclass
class RecoveryConfig:
	alpha_fpr: float
	tpr_target_beta: float
	k_grid: List[int]


class MetaLearner(Protocol):
	def meta_fit(self, tasks: List[TaskData]) -> None:
		...

	def adapt_k_steps(self, task: TaskData, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
		...

	def predict_scores(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		...


class _LogisticHead:
	"""Multi-dimensional logistic detector head.  x @ w + b."""

	def __init__(self, feat_dim: int = 1, device: str = "cpu") -> None:
		self.feat_dim = feat_dim
		self.w = torch.zeros(feat_dim, dtype=torch.float32, device=device)
		self.b = torch.tensor(0.0, dtype=torch.float32, device=device)

	@staticmethod
	def logits(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		# x: (N, D) or (N,), w: (D,) or scalar
		return x @ w + b

	@staticmethod
	def loss(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		logits = _LogisticHead.logits(x, w, b)
		return F.binary_cross_entropy_with_logits(logits, y)


class FOMAMLLearner:
	"""First-order MAML over a logistic head (no second-order graph)."""

	def __init__(
		self,
		inner_lr: float,
		meta_lr: float,
		inner_steps: int,
		outer_iterations: int,
		meta_batch_size_tasks: int,
		seed: int,
		feat_dim: int = 1,
		device: str = "cpu",
	) -> None:
		self.inner_lr = float(inner_lr)
		self.meta_lr = float(meta_lr)
		self.inner_steps = int(inner_steps)
		self.outer_iterations = int(outer_iterations)
		self.meta_batch_size_tasks = int(meta_batch_size_tasks)
		self.rng = np.random.default_rng(seed)
		self.head = _LogisticHead(feat_dim=feat_dim, device=device)
		self.device = device

	def _inner_adapt(self, task: TaskData, k: int, w0: torch.Tensor, b0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		w = w0.clone().detach().requires_grad_(True)
		b = b0.clone().detach().requires_grad_(True)

		for _ in range(k):
			loss = _LogisticHead.loss(task.support_x, task.support_y, w, b)
			gw, gb = torch.autograd.grad(loss, [w, b], create_graph=False)
			w = (w - self.inner_lr * gw).detach().requires_grad_(True)
			b = (b - self.inner_lr * gb).detach().requires_grad_(True)

		return w, b

	def meta_fit(self, tasks: List[TaskData]) -> None:
		if not tasks:
			raise ValueError("meta_fit requires at least one task")

		for _ in range(self.outer_iterations):
			batch_idx = self.rng.choice(len(tasks), size=min(self.meta_batch_size_tasks, len(tasks)), replace=False)
			batch = [tasks[i] for i in batch_idx]

			sum_gw = torch.zeros_like(self.head.w)
			sum_gb = torch.zeros_like(self.head.b)

			for task in batch:
				w_fast, b_fast = self._inner_adapt(task, self.inner_steps, self.head.w, self.head.b)
				q_loss = _LogisticHead.loss(task.query_x, task.query_y, w_fast, b_fast)
				# First-order MAML: use gradient at adapted params and apply to meta-params.
				gw, gb = torch.autograd.grad(q_loss, [w_fast, b_fast], create_graph=False)
				sum_gw += gw.detach()
				sum_gb += gb.detach()

			avg_gw = sum_gw / len(batch)
			avg_gb = sum_gb / len(batch)
			self.head.w = (self.head.w - self.meta_lr * avg_gw).detach()
			self.head.b = (self.head.b - self.meta_lr * avg_gb).detach()

	def adapt_k_steps(self, task: TaskData, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self._inner_adapt(task, k, self.head.w, self.head.b)

	def predict_scores(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		# Score used for fixed-FPR thresholding is detector logit.
		return _LogisticHead.logits(x, w, b).detach()


class ReptileLearner:
	"""Reptile over the same logistic head for fair comparison."""

	def __init__(
		self,
		inner_lr: float,
		meta_lr: float,
		inner_steps: int,
		outer_iterations: int,
		meta_batch_size_tasks: int,
		seed: int,
		feat_dim: int = 1,
		device: str = "cpu",
	) -> None:
		self.inner_lr = float(inner_lr)
		self.meta_lr = float(meta_lr)
		self.inner_steps = int(inner_steps)
		self.outer_iterations = int(outer_iterations)
		self.meta_batch_size_tasks = int(meta_batch_size_tasks)
		self.rng = np.random.default_rng(seed)
		self.head = _LogisticHead(feat_dim=feat_dim, device=device)
		self.device = device

	def _inner_adapt(self, task: TaskData, k: int, w0: torch.Tensor, b0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		w = w0.clone().detach().requires_grad_(True)
		b = b0.clone().detach().requires_grad_(True)

		for _ in range(k):
			loss = _LogisticHead.loss(task.support_x, task.support_y, w, b)
			gw, gb = torch.autograd.grad(loss, [w, b], create_graph=False)
			w = (w - self.inner_lr * gw).detach().requires_grad_(True)
			b = (b - self.inner_lr * gb).detach().requires_grad_(True)

		return w.detach(), b.detach()

	def meta_fit(self, tasks: List[TaskData]) -> None:
		if not tasks:
			raise ValueError("meta_fit requires at least one task")

		for _ in range(self.outer_iterations):
			batch_idx = self.rng.choice(len(tasks), size=min(self.meta_batch_size_tasks, len(tasks)), replace=False)
			batch = [tasks[i] for i in batch_idx]

			target_w = torch.zeros_like(self.head.w)
			target_b = torch.zeros_like(self.head.b)

			for task in batch:
				w_fast, b_fast = self._inner_adapt(task, self.inner_steps, self.head.w, self.head.b)
				target_w += w_fast
				target_b += b_fast

			target_w /= len(batch)
			target_b /= len(batch)

			self.head.w = (self.head.w + self.meta_lr * (target_w - self.head.w)).detach()
			self.head.b = (self.head.b + self.meta_lr * (target_b - self.head.b)).detach()

	def adapt_k_steps(self, task: TaskData, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self._inner_adapt(task, k, self.head.w, self.head.b)

	def predict_scores(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		return _LogisticHead.logits(x, w, b).detach()


def quantile_threshold(clean_scores: np.ndarray, alpha_fpr: float) -> float:
	return float(np.quantile(clean_scores, 1.0 - alpha_fpr))


def compute_tpr_fpr(clean_scores: np.ndarray, wm_scores: np.ndarray, tau: float) -> Tuple[float, float]:
	fpr = float(np.mean(clean_scores > tau))
	tpr = float(np.mean(wm_scores > tau))
	return tpr, fpr


def evaluate_recovery_kstar(
	learner: MetaLearner,
	task: TaskData,
	calib_tau: float,
	cfg: RecoveryConfig,
) -> Dict[str, object]:
	k_metrics: List[Dict[str, float]] = []
	k_star = None
	time_to_target_sec = None
	elapsed_cum = 0.0

	for k in cfg.k_grid:
		t0 = perf_counter()
		w_adapt, b_adapt = learner.adapt_k_steps(task, int(k))
		elapsed = perf_counter() - t0
		elapsed_cum += elapsed

		clean_scores = learner.predict_scores(task.query_x[task.query_y == 0], w_adapt, b_adapt).cpu().numpy()
		wm_scores = learner.predict_scores(task.query_x[task.query_y == 1], w_adapt, b_adapt).cpu().numpy()
		tpr, fpr = compute_tpr_fpr(clean_scores, wm_scores, calib_tau)

		row = {
			"k": int(k),
			"tpr": float(tpr),
			"fpr": float(fpr),
			"latency_ms": float(elapsed * 1000.0),
			"elapsed_cum_sec": float(elapsed_cum),
		}
		k_metrics.append(row)

		if k_star is None and tpr >= cfg.tpr_target_beta and fpr <= cfg.alpha_fpr:
			k_star = int(k)
			time_to_target_sec = float(elapsed_cum)

	return {
		"task_id": task.task_id,
		"k_star": k_star,
		"time_to_target_sec": time_to_target_sec,
		"step_metrics": k_metrics,
	}

