# generator.py
# LSTM-based Generator network
# Generates synthetic price paths designed to MAXIMIZE the hedger's CVaR loss
# This is the adversary in the min-max game

import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, noise_size=8, hidden_size=64, n_layers=2, n_steps=30):
        """
        noise_size: dimension of random noise input at each step
        hidden_size: LSTM memory size
        n_steps: number of trading steps to generate
        """
        super(Generator, self).__init__()
        self.n_steps = n_steps
        self.noise_size = noise_size

        self.lstm = nn.LSTM(
            input_size=noise_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )

        # Output layer: maps LSTM hidden → log return at each step
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        """
        z shape: (batch_size, n_steps, noise_size) — random noise
        returns log_returns: (batch_size, n_steps)
        """
        lstm_out, _ = self.lstm(z)               # (batch, n_steps, hidden)
        log_returns = self.output_layer(lstm_out) # (batch, n_steps, 1)
        return log_returns.squeeze(-1)            # (batch, n_steps)


def generate_paths_from_returns(log_returns, S0=100.0):
    """
    Convert log returns → price paths.
    log_returns: (batch, n_steps) tensor
    returns prices: (batch, n_steps+1) tensor
    """
    batch_size = log_returns.shape[0]
    device = log_returns.device

    # Starting price for all paths
    S0_tensor = torch.full((batch_size, 1), S0, device=device)

    # Cumulative sum of log returns → price multipliers
    cum_returns = torch.cumsum(log_returns, dim=1)  # (batch, n_steps)

    # Price at each step: S0 * exp(cumulative log return)
    prices = S0_tensor * torch.exp(
        torch.cat([torch.zeros(batch_size, 1, device=device), cum_returns], dim=1)
    )  # (batch, n_steps+1)

    return prices


def sample_noise(batch_size, n_steps, noise_size, device):
    """Sample random noise for the generator input."""
    return torch.randn(batch_size, n_steps, noise_size, device=device)


def constrain_log_returns(log_returns, sigma=0.2, dt=1/30, scale=3.0):
    """
    Constrain generator outputs to realistic log return range.
    Without this, the generator produces absurd price paths.
    Max return per step = scale * sigma * sqrt(dt)
    """
    max_val = scale * sigma * (dt ** 0.5)
    return torch.clamp(log_returns, min=-max_val, max=max_val)


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    N_STEPS    = 30
    NOISE_SIZE = 8
    BATCH_SIZE = 4096

    gen = Generator(noise_size=NOISE_SIZE, hidden_size=64,
                    n_layers=2, n_steps=N_STEPS).to(DEVICE)
    print(f"Generator parameters: {sum(p.numel() for p in gen.parameters())}")

    # Quick forward pass test
    z            = sample_noise(BATCH_SIZE, N_STEPS, NOISE_SIZE, DEVICE)
    log_returns  = gen(z)
    log_returns  = constrain_log_returns(log_returns)
    prices       = generate_paths_from_returns(log_returns)

    print(f"Noise shape:       {z.shape}")
    print(f"Log returns shape: {log_returns.shape}")
    print(f"Prices shape:      {prices.shape}")
    print(f"Sample final prices (first 5): {prices[:5, -1].detach().cpu().numpy().round(2)}")
    print("\nGenerator check passed.")