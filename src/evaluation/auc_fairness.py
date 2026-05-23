"""
src/evaluation/auc_fairness.py

AUC-fairness: area under the fairness curve over time.
Two variants:
  - auc_fairness_single_attr : simulation (one sensitive attribute)
  - auc_fairness_all_models  : real dataset (SEX, RACE, AGE)
"""

import numpy as np
import pandas as pd


def _safe_trapz(sub: pd.DataFrame, time_col: str, metric: str,
                min_samples: int) -> float:
    """Normalised trapezoidal integration, excluding unreliable time points."""
    sub = sub.copy().sort_values(time_col)
    if "n_group_min" in sub.columns:
        sub = sub[sub["n_group_min"] >= min_samples]
    sub = sub.dropna(subset=[metric])
    if len(sub) < 3:
        return np.nan
    t = sub[time_col].values.astype(float)
    v = sub[metric].values.astype(float)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)
    return float(np.trapezoid(v, t_norm))


# ── Simulation: single sensitive attribute ────────────────────────────────────

def auc_fairness_single_attr(
    df_dynamic: pd.DataFrame,
    df_pp: pd.DataFrame,
    df_static_agg: pd.DataFrame,
    time_col_dyn: str = "landmark",
    time_col_pp: str = "time",
    min_samples_per_group: int = 20,
) -> pd.DataFrame:
    """
    Compute AUC-fairness for simulation (single sensitive attribute).

    Parameters
    ----------
    df_dynamic    : per-landmark fairness DataFrame (from fairness_metrics loop)
    df_pp         : per-time fairness DataFrame
    df_static_agg : aggregate fairness DataFrame (for M_STATIC scalar value)
    time_col_dyn  : time column in df_dynamic
    time_col_pp   : time column in df_pp
    min_samples_per_group : minimum n_group_min to include a time point

    Returns
    -------
    pd.DataFrame with columns [metric, AUC_M_STATIC, AUC_M_DYNAMIC, AUC_M_PP]
    """
    metrics = ["independence", "separation", "sufficiency"]
    results = []

    for metric in metrics:
        static_row = df_static_agg[df_static_agg["model"] == "M_STATIC"]
        static_val = (
            float(static_row[metric].values[0])
            if len(static_row) > 0 and metric in static_row.columns
            else np.nan
        )
        results.append({
            "metric":        metric,
            "AUC_M_STATIC":  static_val,
            "AUC_M_DYNAMIC": _safe_trapz(df_dynamic, time_col_dyn,
                                          metric, min_samples_per_group),
            "AUC_M_PP":      _safe_trapz(df_pp, time_col_pp,
                                          metric, min_samples_per_group),
        })

    return pd.DataFrame(results)


# ── Real dataset: multiple sensitive attributes ───────────────────────────────

def auc_fairness_all_models(
    df_dynamic: pd.DataFrame,
    df_pp: pd.DataFrame,
    df_static_agg: pd.DataFrame,
    time_col_dyn: str = "landmark",
    time_col_pp: str = "age_lo",
    attrs: list = None,
    min_samples_per_group: int = 50,
) -> pd.DataFrame:
    """
    Compute AUC-fairness for real dataset (multiple sensitive attributes).

    Parameters
    ----------
    df_dynamic    : per-landmark fairness DataFrame with 'attr' column
    df_pp         : per-time fairness DataFrame with 'attr' column
    df_static_agg : aggregate fairness DataFrame with 'attr' column
    time_col_dyn  : time column in df_dynamic
    time_col_pp   : time column in df_pp
    attrs         : list of attribute names (default: ["SEX", "RACE", "AGE"])
    min_samples_per_group : minimum n_group_min to include a time point

    Returns
    -------
    pd.DataFrame with columns [attr, metric, AUC_M_STATIC, AUC_M_DYNAMIC, AUC_M_PP]
    """
    if attrs is None:
        attrs = ["SEX", "RACE", "AGE"]

    metrics = ["independence", "separation", "sufficiency"]
    results = []

    for attr_name in attrs:
        for metric in metrics:
            static_row = df_static_agg[
                (df_static_agg["attr"]  == attr_name) &
                (df_static_agg["model"] == "M_STATIC")
            ]
            static_val = (
                float(static_row[metric].values[0])
                if len(static_row) > 0 and metric in static_row.columns
                else np.nan
            )

            dyn_sub = df_dynamic[df_dynamic["attr"] == attr_name]
            pp_sub  = df_pp[df_pp["attr"] == attr_name]

            results.append({
                "attr":          attr_name,
                "metric":        metric,
                "AUC_M_STATIC":  static_val,
                "AUC_M_DYNAMIC": _safe_trapz(dyn_sub, time_col_dyn,
                                              metric, min_samples_per_group),
                "AUC_M_PP":      _safe_trapz(pp_sub,  time_col_pp,
                                              metric, min_samples_per_group),
            })

    return pd.DataFrame(results)


# ── Bootstrap AUC ────────────────────────────────────────────────────────────

def auc_fairness_bootstrap(df_boot: pd.DataFrame, time_col: str,
                            metric: str, min_reliable: bool = True) -> float:
    """AUC-fairness computed only on reliable bootstrap time points."""
    sub = df_boot.sort_values(time_col).copy()
    if min_reliable and f"{metric}_reliable" in sub.columns:
        sub = sub[sub[f"{metric}_reliable"]]
    sub = sub.dropna(subset=[metric])
    if len(sub) < 3:
        return np.nan
    t = sub[time_col].values.astype(float)
    v = sub[metric].values.astype(float)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)
    return float(np.trapezoid(v, t_norm))
