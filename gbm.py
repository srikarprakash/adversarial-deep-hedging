# gbm.py
# Geometric Brownian Motion simulator
# This generates synthetic stock price paths for training

import numpy as np
import matplotlib.pyplot as plt

def generate_gbm_paths(
    S0=100.0,       # starting stock price
    mu=0.0,         # drift (we use 0 for risk-neutral)
    sigma=0.2,      # volatility (20% is typical for S&P 500)
    T=1.0,          # total time in years (1 year)
    n_steps=30,     # number of trading days (30 steps)
    n_paths=10000,  # number of simulated paths
    seed=42
):
    np.random.seed(seed)
    dt = T / n_steps

    # Random normal shocks
    Z = np.random.normal(0, 1, size=(n_paths, n_steps))

    # GBM formula: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z

    # Build price paths: shape = (n_paths, n_steps+1)
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    for t in range(n_steps):
        prices[:, t + 1] = prices[:, t] * np.exp(log_returns[:, t])

    return prices


def plot_sample_paths(prices, n_show=50):
    plt.figure(figsize=(10, 5))
    plt.plot(prices[:n_show].T, alpha=0.4, linewidth=0.8)
    plt.title("Sample GBM Price Paths")
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.tight_layout()
    plt.savefig("notebooks/gbm_paths.png", dpi=150)
    plt.show()
    print("Plot saved to notebooks/gbm_paths.png")


if __name__ == "__main__":
    prices = generate_gbm_paths()
    print(f"Generated price paths shape: {prices.shape}")
    print(f"Sample final prices (first 5): {prices[:5, -1].round(2)}")
    plot_sample_paths(prices)