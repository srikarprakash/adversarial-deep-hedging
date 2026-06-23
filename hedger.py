# hedger.py
# LSTM-based Hedger network
# At each time step, it looks at the current price and time remaining,
# and outputs a hedge ratio (how many units of stock to hold)

import torch
import torch.nn as nn
import numpy as np


class Hedger(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, n_layers=2):
        """
        input_size=2 because at each step we feed: [normalized_price, time_remaining]
        hidden_size=64: size of LSTM memory
        n_layers=2: two stacked LSTM layers
        """
        super(Hedger, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True  # input shape: (batch, time_steps, features)
        )

        # Final layer maps LSTM output → single hedge ratio
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # hedge ratio between 0 and 1
        )

    def forward(self, x):
        """
        x shape: (batch_size, n_steps, 2)
        returns hedges shape: (batch_size, n_steps)
        """
        lstm_out, _ = self.lstm(x)           # (batch, n_steps, hidden_size)
        hedges = self.output_layer(lstm_out) # (batch, n_steps, 1)
        return hedges.squeeze(-1)            # (batch, n_steps)


def prepare_features(prices, n_steps=30):
    """
    Build the input tensor for the Hedger.
    Features at each time step:
      - normalized price (price / initial price)
      - time remaining (fraction of total time left)
    
    prices shape: (n_paths, n_steps+1)
    returns tensor shape: (n_paths, n_steps, 2)
    """
    n_paths = prices.shape[0]

    # Normalized prices: divide by starting price so everything starts at 1.0
    norm_prices = prices[:, :-1] / prices[:, 0:1]  # (n_paths, n_steps)

    # Time remaining at each step: goes from 1.0 down to 1/n_steps
    time_remaining = np.linspace(1.0, 1.0 / n_steps, n_steps)
    time_remaining = np.tile(time_remaining, (n_paths, 1))  # (n_paths, n_steps)

    # Stack into features: (n_paths, n_steps, 2)
    features = np.stack([norm_prices, time_remaining], axis=-1)

    return torch.tensor(features, dtype=torch.float32)


def cvar_loss(pnl_tensor, alpha=0.1):
    """
    CVaR loss in PyTorch (differentiable so we can backprop through it).
    pnl_tensor shape: (batch_size,)
    """
    sorted_pnl, _ = torch.sort(pnl_tensor)
    cutoff = int(alpha * len(sorted_pnl))
    worst = sorted_pnl[:cutoff]
    return -worst.mean()  # minimize CVaR = make worst cases less bad


def compute_pnl_torch(prices_tensor, hedges_tensor, strike=100.0, tc=0.001):
    """
    Differentiable P&L computation in PyTorch.
    prices_tensor: (batch, n_steps+1)
    hedges_tensor: (batch, n_steps)
    returns pnl: (batch,)
    """
    # Option payoff liability at expiry
    final_prices = prices_tensor[:, -1]
    option_payoff = -torch.clamp(final_prices - strike, min=0.0)

    pnl = torch.zeros(prices_tensor.shape[0], device=prices_tensor.device)

    n_steps = hedges_tensor.shape[1]
    for t in range(n_steps):
        price_change = prices_tensor[:, t + 1] - prices_tensor[:, t]
        delta = hedges_tensor[:, t]

        # Profit from holding delta units
        pnl += delta * price_change

        # Transaction cost
        if t == 0:
            trade = torch.abs(delta)
        else:
            trade = torch.abs(delta - hedges_tensor[:, t - 1])

        pnl -= tc * prices_tensor[:, t] * trade

    pnl += option_payoff
    return pnl


if __name__ == "__main__":
    from gbm import generate_gbm_paths

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate paths
    prices_np = generate_gbm_paths(n_paths=10000)
    prices_tensor = torch.tensor(prices_np, dtype=torch.float32).to(device)

    # Build model
    model = Hedger(input_size=2, hidden_size=64, n_layers=2).to(device)
    print(f"Hedger parameters: {sum(p.numel() for p in model.parameters())}")

    # Quick forward pass test
    features = prepare_features(prices_np).to(device)
    hedges = model(features)
    print(f"Features shape: {features.shape}")
    print(f"Hedges shape:   {hedges.shape}")

    # Quick P&L + CVaR test
    pnl = compute_pnl_torch(prices_tensor, hedges)
    loss = cvar_loss(pnl)
    print(f"Initial CVaR loss (untrained): {loss.item():.4f}")
    print("\nAll checks passed. Ready to train.")