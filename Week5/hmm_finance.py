#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os

# Ensure output directory
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# 1. DOWNLOAD DATA
def download_data(ticker, start="2010-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=False)
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# 2. PREPROCESS (DAILY LOG RETURNS)
def preprocess(df):
    price = df["Adj Close"].astype(float).dropna()
    returns = np.log(price).diff().dropna()
    returns.name = "returns"
    data = pd.concat([price.loc[returns.index], returns], axis=1)
    data.columns = ["price", "returns"]
    return data


# 3. FIT GAUSSIAN HMM
def fit_hmm(returns, n_states=2):
    X = returns.values.reshape(-1, 1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=42
    )
    model.fit(Xs)
    hidden_states = model.predict(Xs)

    # Unscale means
    means_scaled = model.means_.flatten()
    means = scaler.inverse_transform(means_scaled.reshape(-1, 1)).flatten()

    # Unscale variances
    covs_scaled = np.array([model.covars_[i].flatten() for i in range(n_states)])
    variances = covs_scaled.flatten() * (scaler.scale_[0] ** 2)

    return model, hidden_states, means, variances


# 4. PLOT HMM STATES
def plot_states(data, hidden_states, ticker, n_states):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data["price"], label="Price", linewidth=1)

    for s in range(n_states):
        plt.scatter(
            data.index[hidden_states == s],
            data["price"][hidden_states == s],
            s=10,
            label=f"State {s}"
        )

    plt.title(f"{ticker} - HMM States (n_states={n_states})")
    plt.legend()
    outpath = os.path.join(OUT_DIR, f"{ticker}_hmm_states_n{n_states}.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[+] Saved: {outpath}")


# 5. PLOT RETURN DISTRIBUTION
def plot_return_distribution(data, ticker):
    plt.figure(figsize=(8, 4))
    plt.hist(data["returns"], bins=50, density=True, alpha=0.7, color="steelblue")
    plt.title(f"{ticker} Return Distribution")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    outpath = os.path.join(OUT_DIR, f"{ticker}_return_distribution.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[+] Saved: {outpath}")


# MAIN FUNCTION
def run(ticker="AAPL", start="2010-01-01"):
    print(f"Downloading data for {ticker}...")
    df = download_data(ticker, start)

    print("Preprocessing data...")
    data = preprocess(df)

    # Save processed CSV
    processed_csv_path = os.path.join(OUT_DIR, f"{ticker}_processed.csv")
    data.to_csv(processed_csv_path)
    print(f"[+] Processed CSV saved: {processed_csv_path}")

    # Generate return distribution plot
    print("Generating return distribution plot...")
    plot_return_distribution(data, ticker)

    # Fit HMM (2 states)
    print("Fitting HMM (2 states)...")
    model2, states2, means2, variances2 = fit_hmm(data["returns"], n_states=2)
    print("\nMeans (2 states):", means2)
    print("Variances (2 states):", variances2)
    print("Transition Matrix (2 states):\n", model2.transmat_)
    plot_states(data, states2, ticker, 2)

    # Fit HMM (3 states)
    print("\nFitting HMM (3 states)...")
    model3, states3, means3, variances3 = fit_hmm(data["returns"], n_states=3)
    print("\nMeans (3 states):", means3)
    print("Variances (3 states):", variances3)
    print("Transition Matrix (3 states):\n", model3.transmat_)
    plot_states(data, states3, ticker, 3)

    print("\n--- DONE ---")
    print("Generated files (in output/):")
    print(f"- {ticker}_processed.csv")
    print(f"- {ticker}_return_distribution.png")
    print(f"- {ticker}_hmm_states_n2.png")
    print(f"- {ticker}_hmm_states_n3.png")


# RUN
if __name__ == "__main__":
    run("AAPL", "2010-01-01")
