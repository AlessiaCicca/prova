"""
src/data/build_person_period.py

Builds the person-period (discrete-time hazard) dataset from the raw panel.
One row per subject × time-step, up to (but not including) the event.
Target: does the event occur in the NEXT period?
"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def build_person_period(
    df: pd.DataFrame,
    static_cols: list,
    tvc_cols: list,
    trend_cols: list,
    cat_cols: list,
    id_col: str = "ID",
    time_col: str = "Time",
    event_col: str = "Event",
    first_event_col: str = "FirstEventTime",
    sens_col: str = "sens_loan",
    enc_cat: OneHotEncoder = None,
) -> dict:
    """
    Build the person-period dataset from the raw panel.

    Parameters
    ----------
    df              : raw longitudinal panel (one row per subject × time)
    static_cols     : list of static numeric feature column names
    tvc_cols        : list of time-varying covariate column names
    trend_cols      : list of pre-computed trend feature column names
    cat_cols        : list of categorical feature column names
    id_col          : subject ID column name
    time_col        : time column name
    event_col       : binary event indicator column (1 = event at this row)
    first_event_col : column with the first event time per subject (or NaN)
    sens_col        : sensitive attribute column (already merged onto df)
    enc_cat         : fitted OneHotEncoder (from build_static); if None, fit here

    Returns
    -------
    dict with keys:
        X             : feature matrix  (np.float32, shape N × p)
        y             : binary labels   (np.int8,    shape N)  — event_next
        groups        : subject IDs     (shape N)
        sensitive     : sensitive attr  (shape N)
        ages          : loan age per row (shape N)
        enc_cat       : fitted OneHotEncoder
        medians       : pd.Series of column medians
        feature_names : list of feature names matching X columns
    """

    df_pp = df.copy()

    # ── event_next: does the event happen in the next period? ─────────────────
    df_pp["is_event_now"] = (
        df_pp[first_event_col].notna() &
        (df_pp[time_col] == df_pp[first_event_col])
    ).astype(np.int8)

    df_pp["event_next"] = (
        df_pp.groupby(id_col)["is_event_now"]
             .shift(-1)
             .fillna(0)
             .astype(np.int8)
    )

    # ── Drop last row per subject and rows after event ────────────────────────
    last_mask = (
        df_pp[time_col] == df_pp.groupby(id_col)[time_col].transform("max")
    )
    df_pp = df_pp[~last_mask]
    df_pp = df_pp[
        df_pp[first_event_col].isna() |
        (df_pp[time_col] < df_pp[first_event_col])
    ]

    pp_df = df_pp.copy()
    del df_pp
    gc.collect()

    print(
        f"  [person_period] Rows: {len(pp_df):,} | "
        f"event_next=1: {pp_df['event_next'].sum()} "
        f"({pp_df['event_next'].mean():.2%})"
    )

    # ── Categorical encoding ──────────────────────────────────────────────────
    if enc_cat is None:
        enc_cat = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        enc_cat.fit(pp_df[cat_cols])

    cats = enc_cat.transform(pp_df[cat_cols])
    cat_feature_names = list(enc_cat.get_feature_names_out(cat_cols))

    # ── log(1 + age) as temporal feature ─────────────────────────────────────
    log_age = np.log1p(pp_df[time_col].to_numpy(dtype=np.float32)).reshape(-1, 1)

    # ── Numeric imputation ────────────────────────────────────────────────────
    all_num_cols = static_cols + tvc_cols
    medians      = pp_df[all_num_cols].median()

    num = np.hstack([
        pp_df[static_cols].fillna(medians[static_cols]).to_numpy(dtype=np.float32),
        pp_df[tvc_cols].fillna(medians[tvc_cols]).to_numpy(dtype=np.float32),
        pp_df[trend_cols].fillna(0).to_numpy(dtype=np.float32),
    ])

    # ── Assemble X ────────────────────────────────────────────────────────────
    X = np.hstack([num, cats, log_age])

    y         = pp_df["event_next"].to_numpy(dtype=np.int8)
    groups    = pp_df[id_col].to_numpy()
    sensitive = pp_df[sens_col].to_numpy()
    ages      = pp_df[time_col].to_numpy()

    feature_names = static_cols + tvc_cols + trend_cols + cat_feature_names + ["log1p_age"]

    print(f"  [person_period] X shape: {X.shape}  NaN={np.isnan(X).sum()}  Inf={np.isinf(X).sum()}")

    del cats, pp_df
    gc.collect()

    return dict(
        X             = X,
        y             = y,
        groups        = groups,
        sensitive     = sensitive,
        ages          = ages,
        enc_cat       = enc_cat,
        medians       = medians,
        feature_names = feature_names,
    )
