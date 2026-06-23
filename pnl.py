# pnl.py
# Computes the Profit & Loss of a hedging strategy
# P&L = option payoff collected - cost of hedging trades
import numpy as np
def compute_pnl(
    prices,               # shape: (n_paths, n_steps+1)
    hedges,               # shape: (n_paths, n_steps)  <- hedge ratios output by our network
    strike=100.0,         # option strike price (K)
    transaction_cost=0.001  # 0.1% cost per trade (realistic)
):
    n_paths, n_steps = hedges.shape
    # Option payoff at expiry: seller PAYS max(S_T - K, 0)
    # So from hedger's perspective, this is a liability
    final_prices = prices[:, -1]
    option_payoff = -np.maximum(final_prices - strike, 0)  # negative = we owe this
    # Compute trading P&L
    pnl = np.zeros(n_paths)
    for t in range(n_steps):
        price_change = prices[:, t + 1] - prices[:, t]
        delta_hedge = hedges[:, t]
        # Gain from holding delta units of stock
        pnl += delta_hedge * price_change
        # Transaction cost: pay for changing hedge position
        if t == 0:
            trade_size = np.abs(delta_hedge)
        else:
            trade_size = np.abs(delta_hedge - hedges[:, t - 1])
        pnl -= transaction_cost * prices[:, t] * trade_size
    # Add option payoff (the liability we're trying to offset)
    pnl += option_payoff
    return pnl  # shape: (n_paths,)
def compute_cvar(pnl, alpha=0.1):
    """
    CVaR (Conditional Value at Risk) at level alpha.
    = average of the worst alpha% of P&L outcomes.
    We MINIMIZE this (make worst cases less bad).
    """
    sorted_pnl = np.sort(pnl)
    n = len(sorted_pnl)
    cutoff = int(alpha * n)
    cvar = -np.mean(sorted_pnl[:cutoff])  # negative because lower P&L = worse
    return cvar
if __name__ == "__main__":
    from gbm import generate_gbm_paths
    prices = generate_gbm_paths()
    n_paths, n_steps_plus1 = prices.shape
    n_steps = n_steps_plus1 - 1
    # Test with a dummy hedge of 0.5 (always hold 0.5 units of stock)
    dummy_hedges = np.full((n_paths, n_steps), 0.5)
    pnl = compute_pnl(prices, dummy_hedges)
    cvar = compute_cvar(pnl)
    print(f"P&L mean: {pnl.mean():.4f}")
    print(f"P&L std:  {pnl.std():.4f}")
    print(f"CVaR (worst 10%): {cvar:.4f}")