# FairSurvival_CreditRisk


│
├── README.md
├── requirements.txt
├── config.py
│
├── data_generation/
│   ├── simulation/
│   │   ├── simulate_timevarying.R        # timevarying_gnrt.R
│   │   └── simulate_test.R               # testdtv_gnrt.R
│   └── fnma/
│       ├── build_panel.py                # da CodPerformancePanel_optimized
│       ├── build_static.py               # da CodStaticDataset
│       ├── match_hmda.py                 # da CodMatchFreddieHMDA
│       └── notebooks/
│           └── CheckDistributionMatch.ipynb
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── mlp.py                        # classe MLP + init_bias
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── eo_static.py                  # equalized_odds_loss
│   │   └── eo_dynamic.py                 # equalized_odds_loss_dynamic (tutti i mode)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_mlp.py                  # funzione train_mlp
│   │   └── cross_validation.py           # loop GroupKFold + summary
│   ├── data/
│   │   ├── __init__.py
│   │   ├── build_person_period.py        # costruzione dataset PP
│   │   ├── build_dynamic.py              # costruzione dataset landmark
│   │   └── build_static.py              # costruzione dataset statico
│   └── evaluation/
│       ├── __init__.py
│       ├── fairness_metrics.py           # filter_sensitive, fairness_metrics, print_report
│       ├── fairness_plots.py             # plot_fairness_over_time, plot_auc_fairness_bar
│       └── auc_fairness.py              # auc_fairness_single_attr
│
├── experiments/
│   ├── run_simulation.py                 # entry point simulazione
│   ├── run_fnma.py                       # entry point FNMA (quando pronto)
│   └── configs/
│       ├── simulation_fair.yaml          # parametri scenario fair
│       ├── simulation_unfair.yaml        # parametri scenario unfair
│       └── fnma.yaml                     # parametri FNMA
│
├── notebooks/
│   ├── InitialModel.ipynb
│   ├── ModelFairness.ipynb
│   └── Evaluation.ipynb
│
└── outputs/
    ├── simulation/
    │   └── .gitkeep
    └── fnma/
        └── .gitkeep
