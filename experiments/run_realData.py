"""
MAIN RUN for real data analysis

Reads data matched by data_generation/realData/,
builds the two datasets, runs CV, fairness analysis, and grid search.
"""

import argparse
import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import OneHotEncoder
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

# IMPORTS
from config import (
    SEED, DEVICE,
    ALPHA, BETA, 
    EO_MODE_D,
    SCHEDULE_MODE_D, 
    HORIZON_MONTHS, LANDMARKS,
    STATIC_COLS, TVC_COLS, CAT_COLS,
    FAIR_ATTR, GROUP_NAMES,
    N_FOLDS, USE_WANDB, WANDB_ENTITY, WANDB_PROJECT,
    GRID_BETAS, GRID_ALPHAS,
)
from src.data.build_static        import build_static
from src.data.build_dynamic       import build_dynamic
from src.training.cross_validation import run_cv, build_summary_table, find_best_threshold
from src.training.grid_search      import run_grid_search, plot_tradeoff
from src.evaluation.fairness_metrics import (
    fairness_metrics, filter_sensitive, res_to_row,
    print_fairness_report, compute_adTPR_adFPR,
)
from src.evaluation.auc_fairness  import auc_fairness_all_models
from src.evaluation.fairness_plots import (
    plot_separation_over_time, plot_auc_fairness_bar,
)


# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def parse_args():
    p = argparse.ArgumentParser(
        description="Run real-data experiment."
    )
    p.add_argument("--data_path", required=True,
                   help="Path to panel_all_years_sampled.csv")
    p.add_argument("--fair_attr", default="SEX",
                   choices=["SEX", "RACE", "AGE"])
    p.add_argument("--config", default=None,
                   help="Path to YAML config")
    p.add_argument("--grid_search", action="store_true",
                   help="Run grid search after CV")
    p.add_argument("--out_dir", default=None,
                   help="Output directory")
    return p.parse_args()


def load_config(config_path):
    cfg = dict(
        alpha=ALPHA, beta=BETA, 
        eo_mode_d=EO_MODE_D, 
        schedule_mode_d=SCHEDULE_MODE_D, 
        horizon=HORIZON_MONTHS, landmarks=LANDMARKS,
        n_folds=N_FOLDS, use_wandb=USE_WANDB,
        grid_betas=GRID_BETAS, grid_alphas=GRID_ALPHAS,
    )
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        cfg.update(overrides or {})
    return cfg



def _is_default(s):
    num = pd.to_numeric(s, errors="coerce")
    return (num.notna() & (num != 0)).astype(np.int8).values


def _scheduled_balance(orig_upb, r, N, a):
    try:
        orig_upb = float(orig_upb); r = float(r)
        N = int(float(N));          a = float(a)
    except Exception:
        return np.nan
    if any(np.isnan(x) for x in [orig_upb, r, N, a]):
        return np.nan
    if r > 1:
        r /= 100.0
    if r < 0 or N <= 0 or N > 1000:
        return np.nan
    rm = r / 12.0
    if abs(rm) < 1e-10:
        return max(0.0, orig_upb - (orig_upb / N) * a)
    a = np.clip(a, 0, N)
    try:
        num = (1 + rm) ** N - (1 + rm) ** a
        den = (1 + rm) ** N - 1
        return max(0.0, orig_upb * num / den) if den != 0 else np.nan
    except OverflowError:
        return np.nan


# Data loading: load panel, compute features, demographics, and FirstDefaultAge
def load_raw(data_path, fair_attr):
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path, usecols=[
        "loan_sequence_number", "loan_age", "loan_term",
        "current_upb", "current_interest_rate", "estimated_ltv",
        "current_loan_delinquency_status", "loan_amount",
        "original_ltv", "original_dti", "credit_score",
        "interest_rate", "num_borrowers",
        "occupancy_status_orig", "loan_purpose_orig",
        "applicant_sex", "derived_race", "applicant_age",
    ], low_memory=False)
    print(f"  Rows: {len(df):,}  |  Loans: {df['loan_sequence_number'].nunique():,}")

    # Numeric conversions
    for c in ["loan_amount", "interest_rate", "loan_term",
              "loan_age", "current_upb"] + STATIC_COLS + TVC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # BD_pct — balance deviation vs amortisation schedule
    print("  Computing bd_pct...")
    df["b_sched"] = df.apply(
        lambda r: _scheduled_balance(
            r["loan_amount"], r["interest_rate"],
            r["loan_term"],   r["loan_age"]
        ), axis=1
    ).astype("float32")
    df["bd_pct"] = (
        (df["current_upb"] - df["b_sched"]) / df["b_sched"]
    ).replace([np.inf, -np.inf], np.nan).clip(-2, 2).astype("float32")

    # Trend features
    df = df.sort_values(["loan_sequence_number", "loan_age"])
    for col in ["bd_pct", "estimated_ltv", "current_upb"]:
        df[f"{col}_trend"] = (
            df.groupby("loan_sequence_number")[col]
            .transform(lambda x: x - x.shift(2))
        ).clip(-2, 2).fillna(0)

    # Categorical
    df["occupancy_status_orig"] = df["occupancy_status_orig"].astype("category")
    df["loan_purpose_orig"]     = df["loan_purpose_orig"].astype("category")

    # Demographics → binary
    df["sex_bin"]  = df["applicant_sex"].map({1: 0, 2: 1})

    def race_map(x):
        if not isinstance(x, str): return np.nan
        x = x.strip().lower()
        if x in ["white", "asian"]: return 0
        if x in ["black or african american",
                  "american indian or alaska native",
                  "native hawaiian or other pacific islander",
                  "2 or more races", "other"]: return 1
        return np.nan

    df["race_bin"] = df["derived_race"].apply(race_map)
    df["age_bin"]  = df["applicant_age"].map(
        {"<25": 1, "25-34": 0, "35-44": 0, "45-54": 0,
         "55-64": 0, "65-74": 0, ">74": 0}
    )

    for col, name in [("sex_bin",  "sex_bin_loan"),
                       ("race_bin", "race_bin_loan"),
                       ("age_bin",  "age_bin_loan")]:
        per_loan = df.groupby("loan_sequence_number")[col].first().rename(name)
        df = df.merge(per_loan, on="loan_sequence_number", how="left")

    # Default flag → FirstDefaultAge
    is_def = _is_default(df["current_loan_delinquency_status"])
    df["_is_default"] = is_def
    fd_age = (
        df[df["_is_default"] == 1]
        .groupby("loan_sequence_number")["loan_age"].min()
        .rename("FirstDefaultAge")
    )
    df = df.merge(fd_age, on="loan_sequence_number", how="left")
    df.drop(columns=["_is_default",
                      "current_loan_delinquency_status"], inplace=True)

    n_loans = df["loan_sequence_number"].nunique()
    n_def   = df.groupby("loan_sequence_number")["FirstDefaultAge"].first().notna().sum()
    print(f"  Loans: {n_loans:,}  |  Defaulters: {n_def:,} ({n_def/n_loans:.1%})")

    # sensitive attribute column name
    sens_col_map = {
        "SEX":  "sex_bin_loan",
        "RACE": "race_bin_loan",
        "AGE":  "age_bin_loan",
    }
    df["sens_loan"] = df[sens_col_map[fair_attr]]

    return df


#  Fairness analysis 
def run_fairness_analysis(
    y_static, static_oof, sens_by_attr_static,
    y_dynamic, dynamic_oof, sens_by_attr_dynamic, lmk_vals,
    out_dir, cfg):

    attrs = ["SEX", "RACE", "AGE"]

    th_static  = find_best_threshold(y_static,  static_oof)
    th_dynamic = find_best_threshold(y_dynamic, dynamic_oof)
    

    ybin_static  = (static_oof  >= th_static ).astype(int)
    ybin_dynamic = (dynamic_oof >= th_dynamic).astype(int)

    agg_rows = []
    dyn_rows = []
    pp_rows  = []

    for attr_name in attrs:
        group_names = GROUP_NAMES[attr_name]
        s_stat = sens_by_attr_static[attr_name]
        s_dyn  = sens_by_attr_dynamic[attr_name]

        print(f"\n{'='*50}\n  {attr_name}\n{'='*50}")

        # Aggregate
        for mname, y_t, y_p, y_b, sens, th in [
            ("M_STATIC",  y_static,  static_oof,  ybin_static,  s_stat, th_static),
            ("M_DYNAMIC", y_dynamic, dynamic_oof, ybin_dynamic, s_dyn,  th_dynamic),
        ]:
            yt_f, yp_f, sn_f = filter_sensitive(y_t, y_p, sens)
            yb_f = (yp_f >= th).astype(int)
            res  = fairness_metrics(yt_f, yp_f, yb_f, sn_f,
                                    group_names, threshold=th)
            print_fairness_report(mname, res, group_names, label="AGGREGATE")
            agg_rows.append(res_to_row(res, group_names,
                                       {"attr": attr_name, "model": mname}))

        # Dynamic per landmark
        for L in cfg["landmarks"]:
            mask = lmk_vals == L
            if mask.sum() < 100: continue
            yt_f, yp_f, sn_f = filter_sensitive(
                y_dynamic[mask], dynamic_oof[mask], s_dyn[mask]
            )
            if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2: continue
            yb_f = (yp_f >= th_dynamic).astype(int)
            res  = fairness_metrics(yt_f, yp_f, yb_f, sn_f,
                                    group_names, threshold=th_dynamic)
            dyn_rows.append(res_to_row(res, group_names,
                                       {"attr": attr_name,
                                        "model": "M_DYNAMIC",
                                        "landmark": L}))

        # adTPR / adFPR
        print(f"\n  adTPR / adFPR — {attr_name}")
        for mname, y_t, y_b, sens, tpts in [
            ("M_STATIC",  y_static,  ybin_static,  s_stat, None),
            ("M_DYNAMIC", y_dynamic, ybin_dynamic, s_dyn,  lmk_vals),
        ]:
            res = compute_adTPR_adFPR(y_t, y_b, sens, tpts)
            print(f"    {mname:<12} adTPR={res['adTPR']:.4f}  adFPR={res['adFPR']:.4f}")

    df_agg     = pd.DataFrame(agg_rows)
    df_dyn_lmk = pd.DataFrame(dyn_rows)
    df_pp_age  = pd.DataFrame(pp_rows)

    df_agg.to_csv(out_dir / "fairness_aggregate.csv", index=False)
    df_dyn_lmk.to_csv(out_dir / "fairness_dynamic_by_landmark.csv", index=False)
    df_pp_age.to_csv(out_dir / "fairness_pp_by_age.csv", index=False)

    # AUC fairness
    df_auc = auc_fairness_all_models(
        df_dynamic=df_dyn_lmk, df_static_agg=df_agg,
        time_col_dyn="landmark",
        min_samples_per_group=100,
    )
    df_auc.to_csv(out_dir / "auc_fairness_comparison.csv", index=False)
    print("\n=== AUC FAIRNESS ===")
    print(df_auc.to_string(index=False))


    # Plots
    plot_separation_over_time(
        df=df_dyn_lmk, time_col="landmark",
        title="Fairness — M_DYNAMIC by landmark",
        filename="fairness_dynamic_by_landmark.png",
        out_dir=out_dir, static_df=df_agg, min_samples_per_group=100,
    )

    for attr_name in attrs:
        sub = df_auc[df_auc["attr"] == attr_name].drop(columns="attr")
        plot_auc_fairness_bar(
            df_auc=sub, out_dir=out_dir, attr_name=attr_name,
            filename=f"fairness_auc_{attr_name}.png",
        )

    print(f"\nFairness outputs saved in: {out_dir}")



def main():
    args = parse_args()
    cfg  = load_config(args.config)

    out_dir = Path(args.out_dir) if args.out_dir else \
              Path("outputs") / "realData" / args.fair_attr
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = (
        f"realData_{args.fair_attr}"
        f"_S:{cfg['beta']}"
        f"_D:{cfg['alpha']}_{cfg['eo_mode_d']}"
    )

    print(f"\n{'='*60}")
    print(f"  Dataset   :  REAL")
    print(f"  Attr      : {args.fair_attr}")
    print(f"{'='*60}\n")

    # Load raw 
    df = load_raw(args.data_path, args.fair_attr)

    trend_cols = ["bd_pct_trend", "estimated_ltv_trend", "current_upb_trend"]

    enc_cat = OneHotEncoder(handle_unknown="ignore",
                             sparse_output=False, dtype=np.float32)
    enc_cat.fit(df[CAT_COLS])

    # Sensitive arrays for all three attributes (needed for fairness loop)
    sens_col_map = {
        "SEX":  "sex_bin_loan",
        "RACE": "race_bin_loan",
        "AGE":  "age_bin_loan",
    }

    # Build datasets
    print("\nBuilding STATIC dataset...")
    static_data = build_static(
        df=df,
        static_cols=STATIC_COLS, cat_cols=CAT_COLS,
        horizon=cfg["horizon"],
        id_col="loan_sequence_number", time_col="loan_age",
        first_event_col="FirstDefaultAge",
        sens_col="sens_loan", enc_cat=enc_cat,
    )

    print("\nBuilding DYNAMIC dataset...")
    dynamic_data = build_dynamic(
        df=df,
        static_cols=STATIC_COLS, tvc_cols=TVC_COLS,
        cat_cols=CAT_COLS, landmarks=cfg["landmarks"],
        horizon=cfg["horizon"],
        id_col="loan_sequence_number", time_col="loan_age",
        first_event_col="FirstDefaultAge",
        sens_col="sens_loan", enc_cat=enc_cat,
    )

    # Collect sensitive arrays for all attributes
    static_sens_by_attr = {}
    dyn_sens_by_attr    = {}

    for attr_name, col in sens_col_map.items():
        # reindex from original df
        st_ids  = pd.Series(static_data["groups"])
        dy_ids  = pd.Series(dynamic_data["groups"])

        per_loan = df.groupby("loan_sequence_number")[col].first()

        static_sens_by_attr[attr_name] = st_ids.map(per_loan).to_numpy()
        dyn_sens_by_attr[attr_name]    = dy_ids.map(per_loan).to_numpy()

    del df; gc.collect()

    #  CV 
    train_kwargs = dict(
        beta=cfg["beta"], alpha=cfg["alpha"],
        eo_mode_d=cfg["eo_mode_d"], 
        schedule_mode_d=cfg["schedule_mode_d"],
    )

    print("\nTraining M_STATIC...")
    res_static = run_cv(
        X=static_data["X"], y=static_data["y"],
        groups=static_data["groups"], sensitive=static_data["sensitive"],
        model_name="static", n_splits=cfg["n_folds"], **train_kwargs,
    )

    print("\nTraining M_DYNAMIC...")
    res_dynamic = run_cv(
        X=dynamic_data["X"], y=dynamic_data["y"],
        groups=dynamic_data["groups"], sensitive=dynamic_data["sensitive"],
        time_arr=dynamic_data["lmk_vals"], subj_ids=dynamic_data["groups"],
        model_name="dynamic", n_splits=cfg["n_folds"],
        landmarks=cfg["landmarks"], **train_kwargs,
    )


 
    summary = build_summary_table({
        "M_STATIC":  res_static,
        "M_DYNAMIC": res_dynamic,
    })
    print("\n=== CV RESULTS ===")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "cv_results.csv", index=False)


    # Fairness analysis
    print("\n" + "="*60)
    print("FAIRNESS ANALYSIS")
    print("="*60)

    run_fairness_analysis(
        y_static=static_data["y"],
        static_oof=res_static["oof_preds"],
        sens_by_attr_static=static_sens_by_attr,
        y_dynamic=dynamic_data["y"],
        dynamic_oof=res_dynamic["oof_preds"],
        sens_by_attr_dynamic=dyn_sens_by_attr,
        lmk_vals=dynamic_data["lmk_vals"],
        out_dir=out_dir, cfg=cfg,
    )

    # Grid search
    if args.grid_search:
        print("\n" + "="*60)
        print("GRID SEARCH")
        print("="*60)

        df_grid = run_grid_search(
            X_static=static_data["X"], y_static=static_data["y"],
            grp_static=static_data["groups"],
            sens_static=static_data["sensitive"],
            X_dynamic=dynamic_data["X"], y_dynamic=dynamic_data["y"],
            grp_dynamic=dynamic_data["groups"],
            sens_dynamic=dynamic_data["sensitive"],
            lmk_vals=dynamic_data["lmk_vals"],
            group_names=GROUP_NAMES[args.fair_attr],
            betas=cfg["grid_betas"], alphas=cfg["grid_alphas"],
            n_folds=cfg["n_folds"],
            eo_mode_d=cfg["eo_mode_d"],
            out_dir=out_dir, run_tag=run_tag,
        )
        plot_tradeoff(df_grid, out_dir=out_dir, run_tag=run_tag)

    print(f"\nAll outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
