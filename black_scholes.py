# black_scholes.py
# Black-Scholes delta hedge baseline
# This is the classical formula we're comparing against

import numpy as np
from scipy.stats import norm


def bs_delta(S, K, T, sigma=0.2, r=0.0):
    """
    Black-Scholes delta for a European call option.
    = probability that option expires in the money (under risk-neutral measure)
    
    S     : current stock price
    K     : strike price
    T     : time to expiry (in years)
    sigma : volatility
    r     : risk-free rate (0 for simplicity)
    """
    if T <= 0:
        return 1.0 if S > K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_hedge_paths(prices, K=100.0, T=1.0, sigma=0.2, r=0.0):
    """
    Compute Black-Scholes delta hedge ratios for all paths and time steps.
    
    prices : (n_paths, n_steps+1)
    returns hedges : (n_paths, n_steps)
    """
    n_paths, n_steps_plus1 = prices.shape
    n_steps = n_steps_plus1 - 1
    dt = T / n_steps

    hedges = np.zeros((n_paths, n_steps))

    for t in range(n_steps):
        S = prices[:, t]
        time_remaining = T - t * dt
        for i in range(n_paths):
            hedges[i, t] = bs_delta(S[i], K, time_remaining, sigma, r)

    return hedges


def bs_hedge_paths_fast(prices, K=100.0, T=1.0, sigma=0.2, r=0.0):
    """
    Vectorized version — much faster for large n_paths.
    """
    n_paths, n_steps_plus1 = prices.shape
    n_steps = n_steps_plus1 - 1
    dt = T / n_steps

    hedges = np.zeros((n_paths, n_steps))

    for t in range(n_steps):
        S = prices[:, t]                        # (n_paths,)
        time_remaining = T - t * dt

        if time_remaining <= 0:
            hedges[:, t] = (S > K).astype(float)
            continue

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * time_remaining) / \
             (sigma * np.sqrt(time_remaining))
        hedges[:, t] = norm.cdf(d1)

    return hedges


if __name__ == "__main__":
    from gbm import generate_gbm_paths
    from pnl import compute_pnl, compute_cvar

    print("Running Black-Scholes baseline...")
    prices = generate_gbm_paths(n_paths=50000, n_steps=30, seed=1234)

    print("Computing BS delta hedges (vectorized)...")
    hedges = bs_hedge_paths_fast(prices, K=100.0, T=1.0, sigma=0.2)

    pnl  = compute_pnl(prices, hedges, strike=100.0, transaction_cost=0.001)
    cvar = compute_cvar(pnl, alpha=0.1)

    print(f"\nBlack-Scholes Baseline Results:")
    print(f"  P&L mean : {pnl.mean():.4f}")
    print(f"  P&L std  : {pnl.std():.4f}")
    print(f"  CVaR     : {cvar:.4f}")