#!/usr/bin/env python3
"""
LSTM–GARCH Hybrid for Return Forecasting + Regime Detection
----------------------------------------------------------

What this script does
- Trains an LSTM to forecast the next-period log return (conditional mean)
- Fits a GARCH(1,1) on the LSTM residuals to model conditional variance
- Uses a Gaussian HMM on volatility features to infer market regimes (e.g., low-vol vs high-vol)
- Produces forecasts for next-step mean, variance, and regime probabilities

How to use
1) Install deps (Python 3.10+ recommended):
   pip install numpy pandas torch scikit-learn arch hmmlearn matplotlib

2) Prepare a CSV with at least two columns: Date, Close (or Adj Close). Example header:
   Date,Close
   2018-01-02,2683.73
   ...

3) Run:
   python lstm_garch_hybrid.py --csv path/to/prices.csv --date-col Date --price-col Close \
       --lookback 60 --epochs 30 --hidden 64 --layers 1

Notes
- If you don't have data, pass --demo to generate synthetic mean-reverting noisy prices.
- For multiple assets, run per asset or adapt batching logic.
- This is a reference implementation aimed at clarity; tune hyperparams and add regularization for production.
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

# ---------------------------
# Data utilities
# ---------------------------

def generate_demo_prices(n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    # OU-like log-price dynamics + stochastic vol bursts
    logp = np.zeros(n)
    vol = np.zeros(n)
    vol[0] = 0.01
    for t in range(1, n):
        # Volatility mean-reversion with random shocks
        vol[t] = 0.9 * vol[t-1] + 0.1 * 0.012 + rng.normal(0, 0.001)
        vol[t] = max(1e-4, vol[t])
        # Log-price MR around slow drift
        logp[t] = 0.999 * logp[t-1] + 0.0002 + rng.normal(0, vol[t])
    price = np.exp(logp) * 100
    dates = pd.date_range('2010-01-01', periods=n, freq='B')
    return pd.DataFrame({'Date': dates, 'Close': price})


def load_prices(csv_path: str, date_col: str, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{date_col}' and '{price_col}'.")
    df = df[[date_col, price_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df.rename(columns={date_col: 'Date', price_col: 'Close'}, inplace=True)
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Return'] = np.log(df['Close']).diff()
    df.dropna(inplace=True)
    return df

# ---------------------------
# LSTM Model for mean prediction
# ---------------------------

class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take last time-step output
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


@dataclass
class TrainConfig:
    lookback: int = 60
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.0


def build_sequences(returns: np.ndarray, lookback: int):
    X, y = [], []
    for t in range(lookback, len(returns)):
        X.append(returns[t-lookback:t])
        y.append(returns[t])
    X = np.array(X)[:, :, None]  # shape: (N, lookback, 1)
    y = np.array(y)
    return X, y


def train_lstm(train_loader, val_loader, n_features: int, cfg: TrainConfig, device: str = 'cpu'):
    model = LSTMRegressor(n_features=n_features, hidden=cfg.hidden, n_layers=cfg.layers, dropout=cfg.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        print(f"Epoch {epoch:02d} | Train MSE {np.mean(losses):.6f} | Val MSE {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ---------------------------
# GARCH on residuals
# ---------------------------

def fit_garch(residuals: np.ndarray):
    # Student-t innovations are common in finance
    am = arch_model(residuals * 100, mean='Zero', vol='Garch', p=1, q=1, dist='t')
    res = am.fit(disp='off')
    return res

# ---------------------------
# HMM for regime detection
# ---------------------------

def fit_hmm(features: np.ndarray, n_states: int = 2, seed: int = 123):
    hmm = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200, random_state=seed)
    hmm.fit(features)
    states = hmm.predict(features)
    # Sort states by ascending volatility (assume feature includes a volatility proxy)
    # We'll use the feature's second column if available, else the first.
    vol_col = 1 if features.shape[1] > 1 else 0
    state_vol = np.array([features[states == s, vol_col].mean() for s in range(n_states)])
    order = np.argsort(state_vol)
    # Remap states so 0=low vol, 1=..., K-1=high vol
    remap = {old: new for new, old in enumerate(order)}
    remapped = np.vectorize(lambda s: remap[s])(states)
    return hmm, remapped

# ---------------------------
# Main pipeline
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None, help='Path to CSV with Date and Close columns')
    parser.add_argument('--date-col', type=str, default='Date')
    parser.add_argument('--price-col', type=str, default='Close')
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', help='Use synthetic data (ignores --csv)')
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--states', type=int, default=2, help='Number of HMM states')
    args = parser.parse_args()

    if args.demo:
        df = generate_demo_prices()
    else:
        if args.csv is None:
            raise SystemExit("Provide --csv path or use --demo")
        df = load_prices(args.csv, args.date_col, args.price_col)

    df = compute_returns(df)

    # Build sequences
    lookback = args.lookback
    returns = df['Return'].values.astype(np.float64)
    X, y = build_sequences(returns, lookback)

    # Train/val/test splits
    n = len(X)
    n_test = int(n * args.test_split)
    n_val = int((n - n_test) * args.val_split)
    n_train = n - n_test - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)
    test_ds = SeqDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    cfg = TrainConfig(lookback=lookback, batch_size=args.batch, epochs=args.epochs,
                      lr=1e-3, hidden=args.hidden, layers=args.layers, dropout=args.dropout)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_lstm(train_loader, val_loader, n_features=1, cfg=cfg, device=device)

    # In-sample LSTM predictions to compute residuals
    model.eval()
    with torch.no_grad():
        all_pred = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    residuals = y - all_pred

    # Fit GARCH(1,1) on residuals
    garch_res = fit_garch(residuals)
    cond_var = garch_res.conditional_volatility / 100.0  # scale back
    cond_var = np.asarray(cond_var)

    # Regime detection HMM on volatility features
    # Features: [|returns|, garch_vol]
    feat = np.column_stack([np.abs(y), cond_var[-len(y):]])
    hmm, states = fit_hmm(feat, n_states=args.states)

    # Evaluate LSTM mean forecast on test set
    with torch.no_grad():
        y_pred_test = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
    rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))

    # Next-step forecast: last window
    last_window = torch.from_numpy(X[-1:]).float().to(device)
    with torch.no_grad():
        next_mean = float(model(last_window).cpu().numpy()[0])
    # One-step-ahead GARCH forecast
    garch_forecast = garch_res.forecast(horizon=1)
    next_var = float(garch_forecast.variance.values[-1, 0]) / 10000.0  # scale back from percentage^2
    next_vol = math.sqrt(max(1e-12, next_var))

    # Regime probabilities for last point
    last_feat = feat[-1:].astype(np.float64)
    logprob = []
    for s in range(hmm.n_components):
        # Compute posterior p(state=s | obs)
        # Using Bayes with GaussianHMM parameters
        from scipy.stats import multivariate_normal
        mvn = multivariate_normal(mean=hmm.means_[s], cov=hmm.covars_[s])
        prior = hmm.startprob_[s]
        # This is a simplification; for a full posterior, use hmm.predict_proba on a window
        logprob.append(np.log(prior + 1e-12) + mvn.logpdf(last_feat[0]))
    logprob = np.array(logprob)
    probs = np.exp(logprob - logprob.max())
    probs = probs / probs.sum()

    # Map states to vol ordering (0=low vol ... K-1=high vol)
    # Already remapped during fit.

    # Report
    print("\n================ Results ================")
    print(f"Samples: total={len(df)}, sequences={len(X)} | Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")
    print(f"LSTM Test RMSE: {rmse:.6e} (log-returns)")
    print(f"Next-step mean forecast (LSTM): {next_mean:.6e}")
    print(f"Next-step vol forecast (GARCH): {next_vol:.6e}")
    print(f"Latest inferred regime (0=low vol ... {hmm.n_components-1}=high vol): {states[-1]}")
    print("Regime probabilities for latest point:")
    for s, p in enumerate(probs):
        print(f"  State {s}: {p:.3f}")

    # Attach outputs back to dataframe for inspection
    out_df = df.iloc[-len(y):].copy()
    out_df['lstm_pred'] = all_pred[-len(y):]
    out_df['residual'] = residuals
    out_df['garch_vol'] = cond_var[-len(y):]
    out_df['hmm_state'] = states

    out_path = 'lstm_garch_results.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-timestamp outputs to {out_path}")


if __name__ == '__main__':
    main()
