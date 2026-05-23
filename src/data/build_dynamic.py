"""
src/data/build_dynamic.py

Builds the landmark (dynamic) dataset from the raw longitudinal panel.
One row per subject × landmark, features from the snapshot at that landmark.
Target: default in (landmark, landmark + horizon].
"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def build_dynamic(
    df: pd.DataFrame,
    static_cols: list,
    tvc_cols: list,
    cat_cols: list,
    landmarks: list,
    horizon: int,
    id_col: str = "ID",
    time_col: str = "Time",
    first_event_col: str = "FirstEventTime",
    sens_col: str = "sens_loan",
    enc_cat: OneHotEncoder = None,
) -> dict:
    """
    Build the landmark dataset from the raw panel.

    Parameters
    ----------
    df              : raw longitudinal panel (one row per subject × time)
    static_cols     : list of static numeric feature column names
    tvc_cols        : list of time-varying covariate column names
    cat_cols        : list of categorical feature column names
    landmarks       : list of landmark time points (e.g. [1, 2, ..., 12])
    horizon         : prediction horizon in periods
    id_col          : subject ID column name
    time_col        : time column name
    first_event_col : column with the first event time per subject (or NaN)
    sens_col        : sensitive attribute column (already merged onto df)
    enc_cat         : fitted OneHotEncoder (from build_static); if None, fit here

    Returns
    -------
    dict with keys:
        X             : feature matrix  (np.float32, shape N × p)
        y             : binary labels   (np.int8,    shape N)
        groups        : subject IDs     (shape N)
        sensitive     : sensitive attr  (shape N)
        lmk_vals      : landmark value per row (shape N)
        enc_cat       : fitted OneHotEncoder
        enc_lmk       : fitted OneHotEncoder for landmark one-hot
        medians       : pd.Series of column medians
        feature_names : list of feature names matching X columns
    """

    # ── Build one snapshot per landmark ───────────────────────────────────────
    lm_rows = []
    for L in landmarks:
        snap = df[df[time_col] == L].copy()
        if len(snap) == 0:
            continue
        # keep only subjects still at risk at landmark L
        snap = snap[
            snap[first_event_col].isna() | (snap[first_event_col] > L)
        ].copy()
        snap["future_event"] = (
            snap[first_event_col].notna() &
            (snap[first_event_col] > L) &
            (snap[first_event_col] <= L + horizon)
        ).astype(np.int8)
        snap["landmark"] = np.int8(L)
        lm_rows.append(snap)

    landmark_df = pd.concat(lm_rows, ignore_index=True)
    del lm_rows
    gc.collect()

    print(
        f"  [dynamic] Rows: {len(landmark_df):,} | "
        f"Positives: {landmark_df['future_event'].sum()} "
        f"({landmark_df['future_event'].mean():.2%})"
    )

    # ── Categorical encoding ──────────────────────────────────────────────────
    if enc_cat is None:
        enc_cat = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        enc_cat.fit(landmark_df[cat_cols])

    enc_lmk = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=np.float32
    )
    enc_lmk.fit(np.array(landmarks).reshape(-1, 1))

    cats   = enc_cat.transform(landmark_df[cat_cols])
    lmk_oh = enc_lmk.transform(landmark_df[["landmark"]])

    cat_feature_names = list(enc_cat.get_feature_names_out(cat_cols))
    lmk_feature_names = [f"lmk_{L}" for L in landmarks]

    # ── Numeric imputation ────────────────────────────────────────────────────
    all_num_cols = static_cols + tvc_cols
    medians      = landmark_df[all_num_cols].median()

    num = np.hstack([
        landmark_df[static_cols].fillna(medians[static_cols]).to_numpy(dtype=np.float32),
        landmark_df[tvc_cols].fillna(medians[tvc_cols]).to_numpy(dtype=np.float32),
    ])

    # ── Assemble X ────────────────────────────────────────────────────────────
    X = np.hstack([num, cats, lmk_oh])

    y         = landmark_df["future_event"].to_numpy(dtype=np.int8)
    groups    = landmark_df[id_col].to_numpy()
    sensitive = landmark_df[sens_col].to_numpy()
    lmk_vals  = landmark_df["landmark"].to_numpy()

    feature_names = static_cols + tvc_cols + cat_feature_names + lmk_feature_names

    print(f"  [dynamic] X shape: {X.shape}  NaN={np.isnan(X).sum()}  Inf={np.isinf(X).sum()}")

    del cats, lmk_oh, landmark_df
    gc.collect()

    return dict(
        X             = X,
        y             = y,
        groups        = groups,
        sensitive     = sensitive,
        lmk_vals      = lmk_vals,
        enc_cat       = enc_cat,
        enc_lmk       = enc_lmk,
        medians       = medians,
        feature_names = feature_names,
    )
