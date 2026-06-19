#!/usr/bin/env python3
"""Reproduce LR-schedule loss prediction laws for Task 2.

The main experiment fits two published loss-curve models on the cosine
learning-rate curve and evaluates zero-shot prediction on WSD.  The extension
searches all directed transfers among cosine, WSD, and 8-1-1 schedules and
selects the best source-target direction.

References:
  Tissue et al. (2024), "Scaling Law with Learning Rate Annealing"
  Luo et al. (2024/2025), "A Multi-Power Law for Loss Curve Prediction Across
  Learning Rate Schedules"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


LOSS_COL = "Metrics/loss"
VELOCITY_WEIGHT = 0.50


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float
    worst_relative_error: float
    r2: float
    final_abs_error: float
    final_relative_error: float


def load_curves(path: Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_pickle(path)
    curves: dict[str, pd.DataFrame] = {}
    for key, df in raw.items():
        scheduler = key.split("scheduler:", 1)[1].split("_", 1)[0]
        out = df.rename(columns={LOSS_COL: "loss"}).copy()
        out = out[["step", "loss", "lr"]].dropna().sort_values("step")
        curves[scheduler] = out.reset_index(drop=True)
    return curves


def sample_indices(n: int, max_points: int) -> np.ndarray:
    """Dense at the start/end, even in the middle."""
    if n <= max_points:
        return np.arange(n)
    head = np.unique(np.geomspace(1, min(n, 1500), 350).astype(int) - 1)
    tail = n - np.unique(np.geomspace(1, min(n, 4000), 550).astype(int))
    middle_budget = max(max_points - len(head) - len(tail), 200)
    middle = np.linspace(0, n - 1, middle_budget).astype(int)
    return np.unique(np.clip(np.concatenate([head, middle, tail]), 0, n - 1))


def metrics(y: np.ndarray, pred: np.ndarray, start: int = 0) -> Metrics:
    y = y[start:]
    pred = pred[start:]
    err = pred - y
    rel = np.abs(err) / np.maximum(np.abs(y), 1e-12)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return Metrics(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        mape=float(np.mean(rel)),
        worst_relative_error=float(np.max(rel)),
        r2=float(1.0 - ss_res / ss_tot),
        final_abs_error=float(abs(pred[-1] - y[-1])),
        final_relative_error=float(abs(pred[-1] - y[-1]) / max(abs(y[-1]), 1e-12)),
    )


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    display = df[columns].copy()
    for col in display.select_dtypes(include=[np.number]).columns:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, sep, *rows])


def velocity_residual(y: np.ndarray, pred: np.ndarray, idx: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return np.array([], dtype=float)
    v_idx = idx[idx > 0]
    dy = y[v_idx] - y[v_idx - 1]
    dp = pred[v_idx] - pred[v_idx - 1]
    scale = max(float(np.std(dy)), 1e-6)
    return np.sqrt(weight) * (dy - dp) / scale


def tissue_features(lr: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """S1 is cumulative LR; S2 is decayed annealing momentum area."""
    s1 = np.cumsum(lr)
    delta = np.r_[0.0, lr[:-1] - lr[1:]]
    momentum = np.empty_like(lr)
    m = 0.0
    for i, d in enumerate(delta):
        m = rho * m + d
        momentum[i] = m
    s2 = np.cumsum(momentum)
    return np.maximum(s1, 1e-12), s2


def fit_tissue(df: pd.DataFrame, idx: np.ndarray) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    lr = df["lr"].to_numpy(float)
    y = df["loss"].to_numpy(float)
    min_loss = float(y.min())

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float]:
        l0 = min_loss - np.exp(theta[0])
        a = np.exp(theta[1])
        alpha = np.exp(theta[2])
        c = np.exp(theta[3])
        rho = 1.0 / (1.0 + np.exp(-theta[4]))
        return l0, a, alpha, c, rho

    def residual(theta: np.ndarray) -> np.ndarray:
        l0, a, alpha, c, rho = unpack(theta)
        s1, s2 = tissue_features(lr, rho)
        pred = l0 + a * s1 ** (-alpha) - c * s2
        pred = np.maximum(pred[idx], 1e-9)
        return np.log(y[idx]) - np.log(pred)

    init = np.array([np.log(0.08), np.log(0.6), np.log(0.45), np.log(400.0), 2.0])
    res = least_squares(residual, init, loss="huber", f_scale=0.001, max_nfev=2500)
    l0, a, alpha, c, rho = unpack(res.x)
    params = {"L0": l0, "A": a, "alpha": alpha, "C": c, "rho": rho}

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        s1, s2 = tissue_features(new_df["lr"].to_numpy(float), rho)
        return np.maximum(l0 + a * s1 ** (-alpha) - c * s2, 1e-9)

    return params, predict


def fit_tissue_with_velocity_weight(
    df: pd.DataFrame, idx: np.ndarray, velocity_weight: float
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    lr = df["lr"].to_numpy(float)
    y = df["loss"].to_numpy(float)
    min_loss = float(y.min())

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float]:
        l0 = min_loss - np.exp(theta[0])
        a = np.exp(theta[1])
        alpha = np.exp(theta[2])
        c = np.exp(theta[3])
        rho = 1.0 / (1.0 + np.exp(-theta[4]))
        return l0, a, alpha, c, rho

    def predict_from_theta(theta: np.ndarray, new_df: pd.DataFrame) -> np.ndarray:
        l0, a, alpha, c, rho = unpack(theta)
        s1, s2 = tissue_features(new_df["lr"].to_numpy(float), rho)
        return np.maximum(l0 + a * s1 ** (-alpha) - c * s2, 1e-9)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = predict_from_theta(theta, df)
        value_res = np.log(y[idx]) - np.log(pred[idx])
        return np.r_[value_res, velocity_residual(y, pred, idx, velocity_weight)]

    init = np.array([np.log(0.08), np.log(0.6), np.log(0.45), np.log(400.0), 2.0])
    res = least_squares(residual, init, loss="huber", f_scale=0.001, max_nfev=2500)
    l0, a, alpha, c, rho = unpack(res.x)
    params = {"L0": l0, "A": a, "alpha": alpha, "C": c, "rho": rho, "velocity_weight": velocity_weight}

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        return predict_from_theta(res.x, new_df)

    return params, predict


def fit_tissue_velocity_matched(
    df: pd.DataFrame, idx: np.ndarray
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_tissue_with_velocity_weight(df, idx, VELOCITY_WEIGHT)


def mpl_loss_drop_at_indices(lr: np.ndarray, idx: np.ndarray, c: float, beta: float, gamma: float) -> np.ndarray:
    """Luo MPL loss-drop term evaluated at selected indices.

    This follows the MultiPowerLaw implementation/formula:
    LD(t) = sum_k (eta_{k-1}-eta_k)
            * [1 - (1 + C * eta_k^-gamma * S_k(t))^-beta].
    """
    s1 = np.cumsum(lr)
    out = np.zeros(len(idx), dtype=float)
    for j, t in enumerate(idx):
        if t <= 0:
            continue
        lr_k = lr[1 : t + 1]
        lr_gap = lr[:t] - lr[1 : t + 1]
        partial = s1[t] - s1[:t]
        x = np.maximum(lr_k, 1e-12) ** (-gamma) * partial
        g = 1.0 - (1.0 + c * x) ** (-beta)
        out[j] = float(np.sum(lr_gap * g))
    return out


def fit_mpl(df: pd.DataFrame, idx: np.ndarray) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    lr = df["lr"].to_numpy(float)
    y = df["loss"].to_numpy(float)
    s1 = np.maximum(np.cumsum(lr), 1e-12)
    min_loss = float(y.min())

    # Shape parameters from the public MultiPowerLaw README for the 100M model.
    # The official code fits these jointly from several schedules.  This task
    # only allows fitting on one cosine curve, so we keep the MPL shape fixed
    # and fit the curve-specific offset/scale/exponent terms.
    fixed_c, fixed_beta, fixed_gamma = 2.132, 0.598, 0.655

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float]:
        l0 = min_loss - np.exp(theta[0])
        a = np.exp(theta[1])
        alpha = np.exp(theta[2])
        b = np.exp(theta[3])
        return l0, a, alpha, b

    ld_fit = mpl_loss_drop_at_indices(lr, idx, fixed_c, fixed_beta, fixed_gamma)

    def residual(theta: np.ndarray) -> np.ndarray:
        l0, a, alpha, b = unpack(theta)
        pred = l0 + a * s1[idx] ** (-alpha) - b * ld_fit
        pred = np.maximum(pred, 1e-9)
        return np.log(y[idx]) - np.log(pred)

    # These starts are close to the public MPL repository's reported 100M fit,
    # then adapted by the optimizer to this supplied curve.
    starts = [[0.08, 0.6, 0.45, 400.0], [0.05, 0.4, 0.50, 200.0], [0.15, 0.9, 0.35, 800.0]]
    best = None
    for start in starts:
        theta0 = np.log(start)
        res = least_squares(residual, theta0, loss="huber", f_scale=0.001, max_nfev=1200)
        score = float(np.mean(np.abs(res.fun)))
        if best is None or score < best[0]:
            best = (score, res)
    assert best is not None
    l0, a, alpha, b = unpack(best[1].x)
    params = {"L0": l0, "A": a, "alpha": alpha, "B": b, "C": fixed_c, "beta": fixed_beta, "gamma": fixed_gamma}

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        new_lr = new_df["lr"].to_numpy(float)
        new_s1 = np.maximum(np.cumsum(new_lr), 1e-12)
        eval_idx = sample_indices(len(new_lr), 2600)
        if eval_idx[-1] != len(new_lr) - 1:
            eval_idx = np.r_[eval_idx, len(new_lr) - 1]
        ld_eval = mpl_loss_drop_at_indices(new_lr, eval_idx, fixed_c, fixed_beta, fixed_gamma)
        pred_eval = np.maximum(l0 + a * new_s1[eval_idx] ** (-alpha) - b * ld_eval, 1e-9)
        return np.interp(np.arange(len(new_lr)), eval_idx, pred_eval)

    return params, predict


def fit_mpl_with_velocity_weight(
    df: pd.DataFrame, idx: np.ndarray, velocity_weight: float
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    lr = df["lr"].to_numpy(float)
    y = df["loss"].to_numpy(float)
    s1 = np.maximum(np.cumsum(lr), 1e-12)
    min_loss = float(y.min())
    fixed_c, fixed_beta, fixed_gamma = 2.132, 0.598, 0.655
    needed_idx = np.unique(np.r_[idx, idx[idx > 0] - 1])
    ld_needed = mpl_loss_drop_at_indices(lr, needed_idx, fixed_c, fixed_beta, fixed_gamma)
    pos = {int(t): i for i, t in enumerate(needed_idx)}
    idx_pos = np.array([pos[int(t)] for t in idx])

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float]:
        l0 = min_loss - np.exp(theta[0])
        a = np.exp(theta[1])
        alpha = np.exp(theta[2])
        b = np.exp(theta[3])
        return l0, a, alpha, b

    def pred_needed(theta: np.ndarray) -> np.ndarray:
        l0, a, alpha, b = unpack(theta)
        return np.maximum(l0 + a * s1[needed_idx] ** (-alpha) - b * ld_needed, 1e-9)

    def residual(theta: np.ndarray) -> np.ndarray:
        p_needed = pred_needed(theta)
        pred_sparse = np.empty(len(y), dtype=float)
        pred_sparse[needed_idx] = p_needed
        value_res = np.log(y[idx]) - np.log(p_needed[idx_pos])
        return np.r_[value_res, velocity_residual(y, pred_sparse, idx, velocity_weight)]

    starts = [[0.08, 0.6, 0.45, 400.0], [0.05, 0.4, 0.50, 200.0], [0.15, 0.9, 0.35, 800.0]]
    best = None
    for start in starts:
        res = least_squares(residual, np.log(start), loss="huber", f_scale=0.001, max_nfev=1200)
        score = float(np.mean(np.abs(res.fun)))
        if best is None or score < best[0]:
            best = (score, res)
    assert best is not None
    l0, a, alpha, b = unpack(best[1].x)
    params = {
        "L0": l0,
        "A": a,
        "alpha": alpha,
        "B": b,
        "C": fixed_c,
        "beta": fixed_beta,
        "gamma": fixed_gamma,
        "velocity_weight": velocity_weight,
    }

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        new_lr = new_df["lr"].to_numpy(float)
        new_s1 = np.maximum(np.cumsum(new_lr), 1e-12)
        eval_idx = sample_indices(len(new_lr), 2600)
        if eval_idx[-1] != len(new_lr) - 1:
            eval_idx = np.r_[eval_idx, len(new_lr) - 1]
        ld_eval = mpl_loss_drop_at_indices(new_lr, eval_idx, fixed_c, fixed_beta, fixed_gamma)
        pred_eval = np.maximum(l0 + a * new_s1[eval_idx] ** (-alpha) - b * ld_eval, 1e-9)
        return np.interp(np.arange(len(new_lr)), eval_idx, pred_eval)

    return params, predict


def fit_mpl_velocity_matched(
    df: pd.DataFrame, idx: np.ndarray
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_mpl_with_velocity_weight(df, idx, VELOCITY_WEIGHT)


def step_time(df: pd.DataFrame) -> np.ndarray:
    return np.arange(len(df), dtype=float) + 1.0


def effective_time(df: pd.DataFrame, power: float) -> np.ndarray:
    lr = df["lr"].to_numpy(float)
    lr0 = max(float(lr[0]), 1e-12)
    return np.cumsum(np.maximum(lr / lr0, 1e-12) ** power) + 1.0


def schedule_features(df: pd.DataFrame) -> np.ndarray:
    lr = df["lr"].to_numpy(float)
    lr0 = max(float(lr[0]), 1e-12)
    lr_ratio = lr / lr0
    delta_lr = np.r_[0.0, np.diff(lr)] / lr0
    mean_lr = np.cumsum(lr) / (lr0 * (np.arange(len(lr), dtype=float) + 1.0))
    anneal = 1.0 - lr_ratio
    return np.column_stack([lr_ratio, delta_lr, mean_lr, anneal])


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def ridge_predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return design @ coef


def fit_power_model(
    df: pd.DataFrame,
    idx: np.ndarray,
    time_fn: Callable[[pd.DataFrame, float], np.ndarray],
    *,
    learn_time_power: bool = False,
    fixed_time_power: float = 0.0,
    velocity_weight: float = 0.0,
    curvature_weight: float = 0.0,
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    y = df["loss"].to_numpy(float)
    min_loss = float(y.min())

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float]:
        l0 = min_loss - np.exp(theta[0])
        a = np.exp(theta[1])
        alpha = np.exp(theta[2])
        if learn_time_power:
            power = 2.5 / (1.0 + np.exp(-theta[3]))
        else:
            power = fixed_time_power
        return l0, a, alpha, power

    def predict_from_theta(theta: np.ndarray, new_df: pd.DataFrame) -> np.ndarray:
        l0, a, alpha, power = unpack(theta)
        tau = np.maximum(time_fn(new_df, power), 1e-12)
        return np.maximum(l0 + a * tau ** (-alpha), 1e-9)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = predict_from_theta(theta, df)
        value_res = np.log(y[idx]) - np.log(pred[idx])
        extra = []
        if velocity_weight <= 0 and curvature_weight <= 0:
            return value_res
        if velocity_weight > 0:
            v_idx = idx[idx > 0]
            dy = y[v_idx] - y[v_idx - 1]
            dp = pred[v_idx] - pred[v_idx - 1]
            scale = max(float(np.std(dy)), 1e-6)
            extra.append(np.sqrt(velocity_weight) * (dy - dp) / scale)
        if curvature_weight > 0:
            c_idx = idx[idx > 1]
            ddy = y[c_idx] - 2.0 * y[c_idx - 1] + y[c_idx - 2]
            ddp = pred[c_idx] - 2.0 * pred[c_idx - 1] + pred[c_idx - 2]
            scale = max(float(np.std(ddy)), 1e-6)
            extra.append(np.sqrt(curvature_weight) * (ddy - ddp) / scale)
        return np.r_[value_res, *extra]

    init = [np.log(0.08), np.log(0.6), np.log(0.45)]
    if learn_time_power:
        init.append(0.0)
    def safe_residual(theta: np.ndarray) -> np.ndarray:
        return np.nan_to_num(residual(theta), nan=1e6, posinf=1e6, neginf=-1e6)

    res = least_squares(safe_residual, np.array(init), loss="huber", f_scale=0.001, max_nfev=2000)
    l0, a, alpha, power = unpack(res.x)
    params = {
        "L0": l0,
        "A": a,
        "alpha": alpha,
        "time_power": power,
        "velocity_weight": velocity_weight,
        "curvature_weight": curvature_weight,
    }

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        return predict_from_theta(res.x, new_df)

    return params, predict


def fit_step_power(df: pd.DataFrame, idx: np.ndarray) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_power_model(df, idx, lambda new_df, _: step_time(new_df), fixed_time_power=0.0)


def fit_effective_time_power(df: pd.DataFrame, idx: np.ndarray) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_power_model(df, idx, effective_time, learn_time_power=True)


def fit_velocity_matched_effective_time(
    df: pd.DataFrame, idx: np.ndarray
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_power_model(df, idx, effective_time, learn_time_power=True, velocity_weight=VELOCITY_WEIGHT)


def fit_curvature_matched_effective_time(
    df: pd.DataFrame, idx: np.ndarray
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_power_model(
        df,
        idx,
        effective_time,
        learn_time_power=True,
        velocity_weight=0.20,
        curvature_weight=0.05,
    )


def fit_schedule_conditioned_power(
    df: pd.DataFrame, idx: np.ndarray
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    y = df["loss"].to_numpy(float)
    t = step_time(df)
    x_raw = schedule_features(df)
    x, mean, std = standardize_fit(x_raw)
    min_loss = float(y.min())

    def unpack(theta: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        l0 = min_loss - np.exp(theta[0])
        a = np.exp(theta[1])
        alpha = np.exp(theta[2])
        beta = theta[3:]
        return l0, a, alpha, beta

    def predict_arrays(theta: np.ndarray, new_df: pd.DataFrame) -> np.ndarray:
        l0, a, alpha, beta = unpack(theta)
        new_t = step_time(new_df)
        new_x = standardize_apply(schedule_features(new_df), mean, std)
        pred = l0 + a * new_t ** (-alpha) + new_x @ beta
        return np.maximum(pred, 1e-9)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = predict_arrays(theta, df)
        return np.log(y[idx]) - np.log(pred[idx])

    init = np.r_[np.log(0.08), np.log(0.6), np.log(0.45), np.zeros(x.shape[1])]
    res = least_squares(residual, init, loss="huber", f_scale=0.001, max_nfev=2500)
    l0, a, alpha, beta = unpack(res.x)
    params = {
        "L0": l0,
        "A": a,
        "alpha": alpha,
        "feature_names": ["lr_ratio", "delta_lr", "mean_lr", "anneal"],
        "beta": beta.tolist(),
    }

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        return predict_arrays(res.x, new_df)

    return params, predict


def fit_residual_correction(
    df: pd.DataFrame,
    idx: np.ndarray,
    base_fit_fn: Callable[[pd.DataFrame, np.ndarray], tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]],
    method_name: str,
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    base_params, base_predict = base_fit_fn(df, idx)
    y = df["loss"].to_numpy(float)
    base_pred = base_predict(df)
    residual = y - base_pred
    x_raw = schedule_features(df)
    x, mean, std = standardize_fit(x_raw)
    coef = ridge_fit(x[idx], residual[idx], alpha=1e-2)

    params = {
        "base_method": method_name,
        "base_params": base_params,
        "feature_names": ["lr_ratio", "delta_lr", "mean_lr", "anneal"],
        "ridge_coef": coef.tolist(),
    }

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        base = base_predict(new_df)
        new_x = standardize_apply(schedule_features(new_df), mean, std)
        correction = ridge_predict(new_x, coef)
        return np.maximum(base + correction, 1e-9)

    return params, predict


def fit_step_residual_correction(
    df: pd.DataFrame, idx: np.ndarray
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_residual_correction(df, idx, fit_step_power, "StepPower")


def fit_effective_time_residual_correction(
    df: pd.DataFrame, idx: np.ndarray
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_residual_correction(df, idx, fit_effective_time_power, "EffectiveTimePower")


def fit_hybrid_tissue_mpl(df: pd.DataFrame, idx: np.ndarray) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    lr = df["lr"].to_numpy(float)
    y = df["loss"].to_numpy(float)
    min_loss = float(y.min())
    fixed_mpl_c, fixed_beta, fixed_gamma = 2.132, 0.598, 0.655
    ld_fit = mpl_loss_drop_at_indices(lr, idx, fixed_mpl_c, fixed_beta, fixed_gamma)

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float, float]:
        l0 = min_loss - np.exp(theta[0])
        a = np.exp(theta[1])
        alpha = np.exp(theta[2])
        tissue_c = np.exp(theta[3])
        mpl_b = np.exp(theta[4])
        rho = 1.0 / (1.0 + np.exp(-theta[5]))
        return l0, a, alpha, tissue_c, mpl_b, rho

    def residual(theta: np.ndarray) -> np.ndarray:
        l0, a, alpha, tissue_c, mpl_b, rho = unpack(theta)
        s1, s2 = tissue_features(lr, rho)
        pred = l0 + a * s1[idx] ** (-alpha) - tissue_c * s2[idx] - mpl_b * ld_fit
        pred = np.maximum(pred, 1e-9)
        return np.log(y[idx]) - np.log(pred)

    init = np.array([np.log(0.08), np.log(0.6), np.log(0.45), np.log(1e-5), np.log(1e-5), 0.0])
    res = least_squares(residual, init, loss="huber", f_scale=0.001, max_nfev=2500)
    l0, a, alpha, tissue_c, mpl_b, rho = unpack(res.x)
    params = {
        "L0": l0,
        "A": a,
        "alpha": alpha,
        "tissue_C": tissue_c,
        "mpl_B": mpl_b,
        "rho": rho,
        "mpl_C": fixed_mpl_c,
        "beta": fixed_beta,
        "gamma": fixed_gamma,
    }

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        new_lr = new_df["lr"].to_numpy(float)
        s1, s2 = tissue_features(new_lr, rho)
        eval_idx = sample_indices(len(new_lr), 2600)
        if eval_idx[-1] != len(new_lr) - 1:
            eval_idx = np.r_[eval_idx, len(new_lr) - 1]
        ld_eval = mpl_loss_drop_at_indices(new_lr, eval_idx, fixed_mpl_c, fixed_beta, fixed_gamma)
        pred_eval = np.maximum(l0 + a * s1[eval_idx] ** (-alpha) - tissue_c * s2[eval_idx] - mpl_b * ld_eval, 1e-9)
        return np.interp(np.arange(len(new_lr)), eval_idx, pred_eval)

    return params, predict


def fit_power_exponential_with_velocity_weight(
    df: pd.DataFrame, idx: np.ndarray, velocity_weight: float
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    y = df["loss"].to_numpy(float)
    min_loss = float(y.min())
    n_ref = max(len(df), 1)
    t = step_time(df)
    u = t / n_ref

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float]:
        l0 = min_loss - np.exp(theta[0])
        a = np.exp(theta[1])
        alpha = np.exp(theta[2])
        e = np.exp(theta[3])
        lam = np.exp(theta[4])
        return l0, a, alpha, e, lam

    def predict_from_theta(theta: np.ndarray, new_df: pd.DataFrame) -> np.ndarray:
        l0, a, alpha, e, lam = unpack(theta)
        new_t = step_time(new_df)
        new_u = new_t / n_ref
        return np.maximum(l0 + a * new_t ** (-alpha) + e * np.exp(-lam * new_u), 1e-9)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = predict_from_theta(theta, df)
        value_res = np.log(y[idx]) - np.log(pred[idx])
        return np.r_[value_res, velocity_residual(y, pred, idx, velocity_weight)]

    starts = [
        [0.08, 0.6, 0.45, 6.0, 80.0],
        [0.08, 0.6, 0.45, 2.0, 20.0],
        [0.08, 0.6, 0.45, 12.0, 150.0],
    ]
    best = None
    for start in starts:
        res = least_squares(residual, np.log(start), loss="huber", f_scale=0.001, max_nfev=2500)
        score = float(np.mean(np.abs(res.fun)))
        if best is None or score < best[0]:
            best = (score, res)
    assert best is not None
    l0, a, alpha, e, lam = unpack(best[1].x)
    params = {"L0": l0, "A": a, "alpha": alpha, "E": e, "lambda": lam, "velocity_weight": velocity_weight}

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        return predict_from_theta(best[1].x, new_df)

    return params, predict


def fit_power_exponential(df: pd.DataFrame, idx: np.ndarray) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_power_exponential_with_velocity_weight(df, idx, 0.0)


def fit_power_exponential_velocity_matched(
    df: pd.DataFrame, idx: np.ndarray
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    return fit_power_exponential_with_velocity_weight(df, idx, VELOCITY_WEIGHT)


def time_coordinate_matrix(df: pd.DataFrame) -> np.ndarray:
    lr = df["lr"].to_numpy(float)
    lr0 = max(float(lr[0]), 1e-12)
    step = step_time(df)
    lr_time = np.cumsum(np.maximum(lr / lr0, 1e-12))
    lr2_time = np.cumsum(np.maximum(lr / lr0, 1e-12) ** 2.0)
    memory = np.empty_like(lr)
    m = 0.0
    beta = 0.99
    for i, value in enumerate(np.maximum(lr / lr0, 1e-12)):
        m = beta * m + value
        memory[i] = m / (1.0 - beta ** (i + 1))
    memory_time = np.cumsum(memory)
    return np.column_stack([step, lr_time + 1.0, lr2_time + 1.0, memory_time + 1.0])


def fit_time_coordinate_ensemble(
    df: pd.DataFrame, idx: np.ndarray
) -> tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]:
    y = df["loss"].to_numpy(float)
    min_loss = float(y.min())

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        ez = np.exp(z)
        return ez / np.sum(ez)

    def mixed_time(new_df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        mat = time_coordinate_matrix(new_df)
        return np.maximum(mat @ weights, 1e-12)

    def unpack(theta: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        l0 = min_loss - np.exp(theta[0])
        a = np.exp(theta[1])
        alpha = np.exp(theta[2])
        weights = softmax(theta[3:])
        return l0, a, alpha, weights

    def predict_from_theta(theta: np.ndarray, new_df: pd.DataFrame) -> np.ndarray:
        l0, a, alpha, weights = unpack(theta)
        tau = mixed_time(new_df, weights)
        return np.maximum(l0 + a * tau ** (-alpha), 1e-9)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = predict_from_theta(theta, df)
        return np.log(y[idx]) - np.log(pred[idx])

    init = np.r_[np.log(0.08), np.log(0.6), np.log(0.45), np.zeros(4)]
    res = least_squares(residual, init, loss="huber", f_scale=0.001, max_nfev=2500)
    l0, a, alpha, weights = unpack(res.x)
    params = {
        "L0": l0,
        "A": a,
        "alpha": alpha,
        "time_coordinates": ["step", "cum_lr", "cum_lr2", "memory_lr"],
        "weights": weights.tolist(),
    }

    def predict(new_df: pd.DataFrame) -> np.ndarray:
        return predict_from_theta(res.x, new_df)

    return params, predict


def plot_predictions(
    curves: dict[str, pd.DataFrame],
    preds: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
    filename: str,
    title: str,
    targets: list[str],
) -> None:
    fig, axes = plt.subplots(len(targets), 1, figsize=(10, 4 * len(targets)), sharex=False)
    if len(targets) == 1:
        axes = [axes]
    fig.suptitle(title)
    for ax, scheduler in zip(axes, targets):
        df = curves[scheduler]
        x = df["step"].to_numpy()
        ax.plot(x, df["loss"], color="black", lw=1.4, label=f"{scheduler} observed")
        for model_name, by_sched in preds.items():
            ax.plot(x, by_sched[scheduler], lw=1.0, alpha=0.9, label=f"{model_name} predicted")
        ax2 = ax.twinx()
        ax2.plot(x, df["lr"], color="tab:gray", lw=0.7, alpha=0.25)
        ax2.set_ylabel("lr")
        ax.set_title(f"Prediction on {scheduler}")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def plot_transfer_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    compare = summary[summary["role"] == "transfer"].copy()
    compare["direction"] = compare["fit_scheduler"] + " -> " + compare["target"]
    pivot = compare.pivot(index="direction", columns="model", values="mape").sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("mean relative error")
    ax.set_title("All directed transfer errors")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "task2_transfer_comparison.png", dpi=180)
    plt.close(fig)


def plot_transfer_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    transfer = summary[summary["role"] == "transfer"]
    matrix = transfer.pivot_table(index="fit_scheduler", columns="target", values="mape", aggfunc="mean")
    schedulers = ["cosine", "wsd", "811"]
    matrix = matrix.reindex(index=schedulers, columns=schedulers)

    fig, ax = plt.subplots(figsize=(6, 5))
    data = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="viridis_r")
    ax.set_xticks(np.arange(len(schedulers)))
    ax.set_yticks(np.arange(len(schedulers)))
    ax.set_xticklabels(schedulers)
    ax.set_yticklabels(schedulers)
    ax.set_xlabel("target schedule")
    ax.set_ylabel("fit schedule")
    ax.set_title("Average MAPE across models")
    for i in range(len(schedulers)):
        for j in range(len(schedulers)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j] * 100:.2f}%", ha="center", va="center", color="white")
            else:
                ax.text(j, i, "-", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="MAPE")
    fig.tight_layout()
    fig.savefig(out_dir / "task2_transfer_heatmap.png", dpi=180)
    plt.close(fig)


def plot_main_method_comparison(main_summary: pd.DataFrame, out_dir: Path) -> None:
    transfer = main_summary[main_summary["role"] == "transfer"].copy().sort_values("mape")
    colors = ["tab:blue" if not name.startswith("Dev") else "tab:green" for name in transfer["model"]]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(transfer["model"], transfer["mape"] * 100.0, color=colors)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Main task: cosine -> WSD prediction error")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "task2_method_development_comparison.png", dpi=180)
    plt.close(fig)


def is_development_model(name: str) -> bool:
    return name.startswith("Dev")


def filter_prediction_models(
    preds: dict[str, dict[str, np.ndarray]], keep: Callable[[str], bool]
) -> dict[str, dict[str, np.ndarray]]:
    return {name: by_target for name, by_target in preds.items() if keep(name)}


def split_main_extension(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = summary[(summary["fit_scheduler"] == "cosine") & (summary["target"].isin(["cosine", "wsd"]))].copy()
    extension = summary[summary["role"] == "transfer"].copy()
    return main, extension


def summarize_best_transfers(summary: pd.DataFrame, metric: str) -> tuple[dict[str, object], pd.DataFrame]:
    transfer = summary[summary["role"] == "transfer"].copy()
    transfer["direction"] = transfer["fit_scheduler"] + " -> " + transfer["target"]
    per_model = transfer.loc[transfer.groupby("model")[metric].idxmin()].sort_values(metric)
    by_direction = (
        transfer.groupby("direction", as_index=False)
        .agg(
            fit_scheduler=("fit_scheduler", "first"),
            target=("target", "first"),
            mean_mape=("mape", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_final_relative_error=("final_relative_error", "mean"),
        )
        .sort_values(f"mean_{metric}" if metric != "mape" else "mean_mape")
        .reset_index(drop=True)
    )
    best_average = by_direction.iloc[0].to_dict()
    best = {
        "selection_metric": metric,
        "best_direction_average_over_models": best_average,
        "best_direction_per_model": per_model[
            ["model", "direction", "fit_scheduler", "target", "mape", "rmse", "mae", "final_relative_error"]
        ].to_dict(orient="records"),
    }
    return best, by_direction


def run_fit_experiment(
    *,
    curves: dict[str, pd.DataFrame],
    models: dict[str, Callable[[pd.DataFrame, np.ndarray], tuple[dict[str, float], Callable[[pd.DataFrame], np.ndarray]]]],
    fit_scheduler: str,
    target_schedulers: list[str],
    experiment: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    fit_idx = sample_indices(len(curves[fit_scheduler]), args.fit_points)
    fit_idx = fit_idx[fit_idx >= args.eval_start]

    rows: list[dict[str, object]] = []
    params: dict[str, dict[str, float]] = {}
    preds: dict[str, dict[str, np.ndarray]] = {}
    for model_name, fit_fn in models.items():
        preds[model_name] = {}
        print(f"[{experiment}] Fitting {model_name} on {fit_scheduler}...", flush=True)
        model_params, predictor = fit_fn(curves[fit_scheduler], fit_idx)
        params[f"{experiment}:{model_name}"] = model_params
        for target in target_schedulers:
            df = curves[target]
            print(f"  predicting {target}", flush=True)
            pred = predictor(df)
            preds[model_name][target] = pred
            row = asdict(metrics(df["loss"].to_numpy(float), pred, start=args.eval_start))
            if model_name in {
                "Dev4-VelocityMatched",
                "Dev9-TissueVelocity",
                "Dev10-LuoMPLVelocity",
                "Dev11-PowerExpVelocity",
            }:
                row["selected_velocity_weight"] = model_params["velocity_weight"]
            row.update(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "fit_scheduler": fit_scheduler,
                    "target": target,
                    "role": "fit_check" if target == fit_scheduler else "transfer",
                }
            )
            rows.append(row)
    return rows, params, preds


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curves(Path(args.pkl))

    models = {
        "Tissue2024": fit_tissue,
        "LuoMPL": fit_mpl,
        "Dev1-EffectiveTime": fit_effective_time_power,
        "Dev2-StepResidual": fit_step_residual_correction,
        "Dev3-EffectiveResidual": fit_effective_time_residual_correction,
        "Dev4-VelocityMatched": fit_velocity_matched_effective_time,
        "Dev5-HybridTissueMPL": fit_hybrid_tissue_mpl,
        "Dev6-CurvatureMatched": fit_curvature_matched_effective_time,
        "Dev7-PowerExponential": fit_power_exponential,
        "Dev8-TimeEnsemble": fit_time_coordinate_ensemble,
        "Dev9-TissueVelocity": fit_tissue_velocity_matched,
        "Dev10-LuoMPLVelocity": fit_mpl_velocity_matched,
        "Dev11-PowerExpVelocity": fit_power_exponential_velocity_matched,
    }

    rows: list[dict[str, object]] = []
    params: dict[str, dict[str, float]] = {}

    schedulers = ["cosine", "wsd", "811"]
    preds_by_fit: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for fit_scheduler in schedulers:
        target_schedulers = schedulers
        exp_rows, exp_params, exp_preds = run_fit_experiment(
            curves=curves,
            models=models,
            fit_scheduler=fit_scheduler,
            target_schedulers=target_schedulers,
            experiment=f"{fit_scheduler}_fit",
            args=args,
        )
        rows.extend(exp_rows)
        params.update(exp_params)
        preds_by_fit[fit_scheduler] = exp_preds

    summary = pd.DataFrame(rows)
    summary["direction"] = summary["fit_scheduler"] + " -> " + summary["target"]
    summary.to_csv(out_dir / "task2_metrics.csv", index=False)
    baseline_summary = summary[~summary["model"].map(is_development_model)].copy()
    development_summary, _ = split_main_extension(summary)
    main_summary, extension_summary = split_main_extension(baseline_summary)
    main_summary.to_csv(out_dir / "task2_main_metrics.csv", index=False)
    extension_summary.to_csv(out_dir / "task2_extension_metrics.csv", index=False)
    development_summary.to_csv(out_dir / "task2_development_metrics.csv", index=False)
    with open(out_dir / "task2_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    best, transfer_ranking = summarize_best_transfers(baseline_summary, args.selection_metric)
    transfer_ranking.to_csv(out_dir / "task2_transfer_ranking.csv", index=False)
    with open(out_dir / "task2_best_transfer.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print("Writing plots...", flush=True)
    for fit_scheduler, preds in preds_by_fit.items():
        plot_predictions(
            curves,
            filter_prediction_models(preds, lambda name: not is_development_model(name)),
            out_dir,
            f"task2_{fit_scheduler}_fit_predictions.png",
            f"Fit {fit_scheduler}, predict cosine / WSD / 8-1-1",
            schedulers,
        )
    plot_predictions(
        curves,
        preds_by_fit["cosine"],
        out_dir,
        "task2_method_development_predictions.png",
        "Method development: fit cosine, predict cosine / WSD / 8-1-1",
        schedulers,
    )
    plot_transfer_comparison(baseline_summary, out_dir)
    plot_transfer_heatmap(baseline_summary, out_dir)
    plot_main_method_comparison(development_summary, out_dir)

    print("Writing markdown report...", flush=True)
    metric_cols = [
        "experiment",
        "model",
        "fit_scheduler",
        "target",
        "role",
        "direction",
        "mae",
        "rmse",
        "mape",
        "r2",
        "final_abs_error",
        "final_relative_error",
        "selected_velocity_weight",
    ]
    best_avg = best["best_direction_average_over_models"]
    per_model_best = pd.DataFrame(best["best_direction_per_model"])
    main_transfer = main_summary[main_summary["role"] == "transfer"]
    development_transfer = development_summary[development_summary["role"] == "transfer"]
    proposed_transfer = development_transfer[development_transfer["model"].map(is_development_model)]
    baseline_transfer = development_transfer[~development_transfer["model"].map(is_development_model)]
    main_conclusion_lines = [
        "## Method Development Conclusion",
        "",
        f"- The reproduced baselines reach average MAPE {baseline_transfer['mape'].mean():.4%} on the requested `cosine -> WSD` transfer.",
        f"- The proposed methods reach average MAPE {proposed_transfer['mape'].mean():.4%} on the same transfer.",
        f"- Best reproduced baseline on WSD is `{baseline_transfer.loc[baseline_transfer['mape'].idxmin(), 'model']}` with MAPE {baseline_transfer['mape'].min():.4%}.",
        f"- Best proposed method on WSD is `{proposed_transfer.loc[proposed_transfer['mape'].idxmin(), 'model']}` with MAPE {proposed_transfer['mape'].min():.4%}.",
        f"- Best overall method on WSD is `{development_transfer.loc[development_transfer['mape'].idxmin(), 'model']}` with MAPE {development_transfer['mape'].min():.4%}.",
    ]
    extension_conclusion_lines = [
        "## Extension Conclusion",
        "",
        f"- Best average transfer direction by `{args.selection_metric}` is `{best_avg['direction']}` with mean MAPE {best_avg['mean_mape']:.4%}.",
        "- Per-model winners are:",
        markdown_table(
            per_model_best,
            ["model", "direction", "mape", "rmse", "mae", "final_relative_error"],
        ),
    ]
    report_lines = [
        "# Task 2 Reproduction Results",
        "",
        "This report separates the required reproduction task, the transfer extension, and the method-development experiments.",
        "",
        "# Part 1: Required Reproduction",
        "",
        "Fit on the cosine LR schedule and test on the WSD LR schedule. The `cosine -> cosine` rows are fit-quality checks; the `cosine -> WSD` rows are the requested cross-schedule prediction results.",
        "",
        markdown_table(main_summary, metric_cols),
        "",
        "## Reproduction Conclusion",
        "",
        f"- The reproduced baselines reach average MAPE {main_transfer['mape'].mean():.4%} on the requested `cosine -> WSD` transfer.",
        f"- Best reproduced baseline on WSD is `{main_transfer.loc[main_transfer['mape'].idxmin(), 'model']}` with MAPE {main_transfer['mape'].min():.4%}.",
        "",
        "# Part 2: Extension",
        "",
        "Evaluate all six directed transfers among `cosine`, `WSD`, and `8-1-1`, then choose the best direction by the selected metric.",
        "",
        "## Extension Transfer Metrics",
        "",
        markdown_table(extension_summary.sort_values(["fit_scheduler", "target", "model"]), metric_cols),
        "",
        "## Extension Transfer Ranking",
        "",
        markdown_table(
            transfer_ranking,
            ["direction", "fit_scheduler", "target", "mean_mape", "mean_rmse", "mean_mae", "mean_final_relative_error"],
        ),
        "",
        *extension_conclusion_lines,
        "",
        "# Part 3: Method Development",
        "",
        "Compare the reproduced baselines with the proposed loss-curve fitting strategies on the original `cosine -> WSD` task.",
        "",
        markdown_table(development_summary, metric_cols),
        "",
        *main_conclusion_lines,
        "",
        "## Notes",
        "",
        "- `Tissue2024` uses `L = L0 + A*S1^{-alpha} - C*S2`, with `S2` implemented as decayed LR-annealing momentum.",
        "- `LuoMPL` follows the public MultiPowerLaw repository formula `L = L0 + A*S1^{-alpha} - B*LD`, using the repository's 100M shape parameters for `C`, `beta`, and `gamma` and fitting the remaining curve-specific parameters on the chosen source schedule.",
        "- Main reproduction outputs: `task2_main_metrics.csv`, `task2_cosine_fit_predictions.png`.",
        "- Extension outputs: `task2_extension_metrics.csv`, `task2_transfer_ranking.csv`, `task2_best_transfer.json`, `task2_transfer_comparison.png`, `task2_transfer_heatmap.png`, `task2_wsd_fit_predictions.png`, `task2_811_fit_predictions.png`.",
        "- Method-development outputs: `task2_development_metrics.csv`, `task2_method_development_predictions.png`, `task2_method_development_comparison.png`.",
        "- Combined/debug outputs: `task2_metrics.csv`, `task2_params.json`.",
    ]
    (out_dir / "task2_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="loss curves/gpt_loss+lrs.pkl")
    parser.add_argument("--out-dir", default="task2_outputs")
    parser.add_argument("--fit-points", type=int, default=2200)
    parser.add_argument("--eval-start", type=int, default=500)
    parser.add_argument("--selection-metric", choices=["mape", "rmse", "mae"], default="mape")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
