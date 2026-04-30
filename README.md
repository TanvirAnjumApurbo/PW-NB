# PW-NB: Proximity-Weighted Naive Bayes

A Naive Bayes classifier that uses **Proximal Ratio (PR)** scores from Amer et al. (2025) to weight training instances during parameter estimation. Points with higher local class consistency contribute more to class-conditional density estimates.

## Project Structure

```
PW-NB/
├── src/
│   ├── proximal_ratio.py   # PR computation (Equations 1-4)
│   ├── pw_nb.py             # GaussianPWNB & MultinomialPWNB classifiers
│   ├── metrics.py           # 10 evaluation metrics including ECE, Brier
│   ├── datasets.py          # 57 benchmark datasets (OpenML, CSV-driven DIDs)
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

# Full experiment (57 datasets, auto-resumes)
python experiments/run_experiment.py

# Statistical tests & figures
python experiments/statistical_tests.py
python experiments/visualize.py
```

## Key Results

- **PW-NB(auto) ranks 1st on 7/10 metrics** across 57 datasets (avg rank 3.75–3.99)
- **33 wins, 3 ties, 21 losses** vs GaussianNB on accuracy (58% win rate)
- All Friedman tests significant (p < 0.05) across all 10 metrics
- Robust to choice of k; auto-selection via inner CV adds a consistent edge

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
