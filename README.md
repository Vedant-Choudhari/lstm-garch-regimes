# LSTM–GARCH Hybrid with Regime Detection

A reference implementation that:
- Trains an **LSTM** to forecast next-step log returns (conditional mean).
- Fits **GARCH(1,1)** on LSTM residuals for conditional variance (volatility) forecasts.
- Runs a **Gaussian HMM** on volatility features to infer **market regimes** (e.g., low vs high vol).

## Quickstart

```bash
# Create and activate a virtual env (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run on synthetic data
python src/lstm_garch_hybrid.py --demo

# Or run on your CSV
python src/lstm_garch_hybrid.py --csv examples/prices_demo.csv --date-col Date --price-col Close   --lookback 60 --epochs 30 --hidden 64 --layers 1
```

The script prints test RMSE, next-step mean/vol forecasts, latest regime, and saves
`lstm_garch_results.csv` with columns: `Return`, `lstm_pred`, `residual`, `garch_vol`, `hmm_state`.

## Repository Structure

```
lstm-garch-regime/
├─ src/
│  └─ lstm_garch_hybrid.py
├─ examples/
│  └─ prices_demo.csv
├─ scripts/
│  └─ train_demo.sh
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

## Create and push a Git repository

```bash
git init
git add .
git commit -m "Initial commit: LSTM–GARCH hybrid with regime detection"
# Create a new repo on GitHub/GitLab/Bitbucket, then:
git branch -M main
git remote add origin <YOUR_REMOTE_URL>
git push -u origin main
```

## Notes

- This is a reference implementation aimed at clarity; for production use, consider hyperparameter tuning,
  walk-forward validation, regularization, model monitoring, and robust error handling.
- For multi-asset panels, you can loop assets or extend batching to include asset IDs and embeddings.
