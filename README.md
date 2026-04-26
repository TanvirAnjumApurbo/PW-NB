# PW-NB: Proximity-Weighted Naive Bayes

A Naive Bayes classifier that uses **Proximal Ratio (PR)** scores from Amer et al. (2025) to weight training instances during parameter estimation. Points with higher local class consistency contribute more to class-conditional density estimates.

## Project Structure

```
PW-NB/
├── src/
│   ├── proximal_ratio.py   # PR computation (Equations 1-4)
│   ├── pw_nb.py             # GaussianPWNB & MultinomialPWNB classifiers
│   ├── metrics.py           # 10 evaluation metrics including ECE, Brier
│   ├── datasets.py          # 24 benchmark datasets (sklearn + OpenML)
│   ├── baselines.py         # NB baseline registry
│   └── utils.py             # Logging, seeding
├── experiments/
│   ├── run_experiment.py    # Main experiment runner (10-fold CV)
│   ├── statistical_tests.py # Friedman, Wilcoxon, rank computation
│   └── visualize.py         # CD diagrams, bar charts, PR distributions
├── tests/
│   ├── test_proximal_ratio.py  # 23 tests for PR computation
│   ├── test_pw_nb.py           # 131 tests (sklearn API compliance)
│   └── test_metrics.py         # 7 metric tests
├── docs/
│   └── PR_understanding.md  # Paper analysis & PR algorithm walkthrough
├── results/                 # Experiment outputs (CSV, figures, stats)
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Quick experiment (3 datasets)
python experiments/run_experiment.py --quick

# Full experiment (23 datasets)
python experiments/run_experiment.py --datasets all

# Statistical tests & figures
python experiments/statistical_tests.py
python experiments/visualize.py
```

## Key Results

- PW-NB variants rank **1st-4th** across 23 datasets on accuracy (avg rank 3.43-3.96)
- **16 wins, 4 ties, 3 losses** vs GaussianNB on accuracy
- All Friedman tests significant (p < 0.05) across 10 metrics
- Robust to choice of k (no significant differences among k=5,15,30,45)

See `results/REPORT.md` for the full experimental report.

## sklearn Compatibility

Both `GaussianPWNB` and `MultinomialPWNB` are fully sklearn-compatible:

```python
from src.pw_nb import GaussianPWNB

clf = GaussianPWNB(k=15)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
```

## Reference

Amer, A. et al. (2025). "Proximity Ratio-Based k-Nearest Neighbor Algorithm for Classification."
