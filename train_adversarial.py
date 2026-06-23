# train_adversarial.py
# The min-max adversarial training loop
# Hedger minimizes CVaR, Generator maximizes CVaR
# They alternate updates — this is the core of the paper

import torch
import numpy as np
import matplotlib.pyplot as plt

from hedger import Hedger, prepare_features, cvar_loss, compute_pnl_torch
from generator import (Generator, generate_paths_from_returns,
                       sample_noise, constrain_log_returns)

# ── Config ───────────────────────────────────────────────────────────────────
N_STEPS      = 30
NOISE_SIZE   = 8
HIDDEN_SIZE  = 64
N_LAYERS     = 2
BATCH_SIZE   = 2048
N_EPOCHS     = 300
LR_HEDGER    = 1e-3
LR_GEN       = 1e-3
N_HEDGER_STEPS = 3   # how many hedger updates per generator update
STRIKE       = 100.0
TC           = 0.001
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")
print(f"Min-max game: Hedger vs Generator")
print(f"Epochs: {N_EPOCHS} | Batch: {BATCH_SIZE}\n")

# ── Models ───────────────────────────────────────────────────────────────────
hedger = Hedger(input_size=2, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS).to(DEVICE)
gen    = Generator(noise_size=NOISE_SIZE, hidden_size=HIDDEN_SIZE,
                   n_layers=N_LAYERS, n_steps=N_STEPS).to(DEVICE)

# Load pretrained hedger weights (warm start — much faster convergence)
hedger.load_state_dict(torch.load("models/hedger.pth", map_location=DEVICE))
print("Loaded pretrained hedger weights for warm start.")

opt_hedger = torch.optim.Adam(hedger.parameters(), lr=LR_HEDGER)
opt_gen    = torch.optim.Adam(gen.parameters(),    lr=LR_GEN)

sched_hedger = torch.optim.lr_scheduler.StepLR(opt_hedger, step_size=100, gamma=0.5)
sched_gen    = torch.optim.lr_scheduler.StepLR(opt_gen,    step_size=100, gamma=0.5)

# ── Helper: generate a batch of paths from Generator ─────────────────────────
def gen_batch(batch_size):
    z           = sample_noise(batch_size, N_STEPS, NOISE_SIZE, DEVICE)
    log_returns = gen(z)
    log_returns = constrain_log_returns(log_returns)
    prices      = generate_paths_from_returns(log_returns, S0=STRIKE)
    return prices  # (batch, n_steps+1)

# ── Helper: build features tensor from prices tensor ─────────────────────────
def prices_to_features(prices_t):
    """prices_t: (batch, n_steps+1) tensor on DEVICE"""
    prices_np   = prices_t.detach().cpu().numpy()
    features_np = prepare_features(prices_np, n_steps=N_STEPS)
    return features_np.to(DEVICE)

# ── Training Loop ─────────────────────────────────────────────────────────────
hedger_losses = []
gen_losses    = []

for epoch in range(1, N_EPOCHS + 1):

    hedger.train()
    gen.train()

    # ── Phase 1: Train Hedger (minimize CVaR) ──────────────────────────────
    # Generator is frozen, Hedger updates N_HEDGER_STEPS times
    for _ in range(N_HEDGER_STEPS):
        with torch.no_grad():
            prices_t = gen_batch(BATCH_SIZE)  # generator frozen here

        features_t = prices_to_features(prices_t)

        opt_hedger.zero_grad()
        hedges  = hedger(features_t)
        pnl     = compute_pnl_torch(prices_t, hedges, strike=STRIKE, tc=TC)
        h_loss  = cvar_loss(pnl, alpha=0.1)
        h_loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger.parameters(), max_norm=1.0)
        opt_hedger.step()

    # ── Phase 2: Train Generator (maximize CVaR) ───────────────────────────
    # Hedger is frozen, Generator updates once
    opt_gen.zero_grad()

    prices_t   = gen_batch(BATCH_SIZE)        # generator active here
    features_t = prices_to_features(prices_t)

    with torch.no_grad():
        hedges = hedger(features_t)           # hedger frozen here

    # Recompute pnl with generator graph intact
    pnl    = compute_pnl_torch(prices_t, hedges.detach(), strike=STRIKE, tc=TC)
    g_loss = -cvar_loss(pnl, alpha=0.1)      # negative = maximize CVaR
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
    opt_gen.step()

    sched_hedger.step()
    sched_gen.step()

    hedger_losses.append(h_loss.item())
    gen_losses.append(-g_loss.item())  # store as positive CVaR

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}/{N_EPOCHS} | "
              f"Hedger CVaR: {h_loss.item():.4f} | "
              f"Generator CVaR: {-g_loss.item():.4f}")

# ── Save Models ───────────────────────────────────────────────────────────────
torch.save(hedger.state_dict(), "models/adversarial_hedger.pth")
torch.save(gen.state_dict(),    "models/adversarial_generator.pth")
print("\nModels saved.")

# ── Plot Training Curves ──────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(hedger_losses, label="Hedger CVaR (minimizing)", color="steelblue", linewidth=1.2)
plt.plot(gen_losses,    label="Generator CVaR (maximizing)", color="tomato",    linewidth=1.2)
plt.title("Adversarial Training: Min-Max Game")
plt.xlabel("Epoch")
plt.ylabel("CVaR")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/adversarial_training_curve.png", dpi=150)
plt.show()
print("Adversarial training curve saved.")

# ── Final Evaluation ──────────────────────────────────────────────────────────
print("\n── Final Evaluation ──")
hedger.eval()
gen.eval()

with torch.no_grad():
    # Evaluate on Generator's worst-case paths
    prices_adv  = gen_batch(50000)
    features_adv = prices_to_features(prices_adv)
    hedges_adv  = hedger(features_adv)
    pnl_adv     = compute_pnl_torch(prices_adv, hedges_adv, strike=STRIKE, tc=TC)
    pnl_adv_np  = pnl_adv.cpu().numpy()

    # Also evaluate on standard GBM paths (for comparison)
    from gbm import generate_gbm_paths
    prices_gbm_np  = generate_gbm_paths(n_paths=50000, n_steps=N_STEPS, seed=1234)
    prices_gbm_t   = torch.tensor(prices_gbm_np, dtype=torch.float32).to(DEVICE)
    features_gbm_t = prepare_features(prices_gbm_np, n_steps=N_STEPS).to(DEVICE)
    hedges_gbm     = hedger(features_gbm_t)
    pnl_gbm        = compute_pnl_torch(prices_gbm_t, hedges_gbm, strike=STRIKE, tc=TC)
    pnl_gbm_np     = pnl_gbm.cpu().numpy()

cvar_adv = -np.percentile(pnl_adv_np, 10)
cvar_gbm = -np.percentile(pnl_gbm_np, 10)

print(f"CVaR on adversarial paths: {cvar_adv:.4f}")
print(f"CVaR on GBM paths:         {cvar_gbm:.4f}")
print(f"P&L mean (adversarial):    {pnl_adv_np.mean():.4f}")
print(f"P&L std  (adversarial):    {pnl_adv_np.std():.4f}")

# ── Comparison Plot ───────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.hist(pnl_gbm_np, bins=100, alpha=0.5, color="steelblue",
         label=f"GBM paths (CVaR={cvar_gbm:.2f})", density=True)
plt.hist(pnl_adv_np, bins=100, alpha=0.5, color="tomato",
         label=f"Adversarial paths (CVaR={cvar_adv:.2f})", density=True)
plt.axvline(np.percentile(pnl_gbm_np, 10), color="steelblue",
            linestyle="--", linewidth=1.5)
plt.axvline(np.percentile(pnl_adv_np, 10), color="tomato",
            linestyle="--", linewidth=1.5)
plt.title("P&L Distribution: GBM vs Adversarial Market Scenarios")
plt.xlabel("P&L")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/adversarial_vs_gbm.png", dpi=150)
plt.show()
print("Comparison plot saved to notebooks/adversarial_vs_gbm.png")