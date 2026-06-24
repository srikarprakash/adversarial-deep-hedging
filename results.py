# results.py
# Final comparison plot: Black-Scholes vs Deep Hedger vs Adversarial Hedger
# This is the key result figure of the project

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gbm import generate_gbm_paths
from hedger import Hedger, prepare_features, compute_pnl_torch
from generator import Generator, generate_paths_from_returns, sample_noise, constrain_log_returns
from black_scholes import bs_hedge_paths_fast
from pnl import compute_pnl, compute_cvar

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS  = 30
N_PATHS  = 50000
STRIKE   = 100.0
TC       = 0.001
NOISE_SIZE = 8

print(f"Generating final results on {N_PATHS} paths...")

# ── Load Models ───────────────────────────────────────────────────────────────
hedger = Hedger(input_size=2, hidden_size=64, n_layers=2).to(DEVICE)
hedger.load_state_dict(torch.load("models/adversarial_hedger.pth", map_location=DEVICE))
hedger.eval()

gen = Generator(noise_size=NOISE_SIZE, hidden_size=64, n_layers=2, n_steps=N_STEPS).to(DEVICE)
gen.load_state_dict(torch.load("models/adversarial_generator.pth", map_location=DEVICE))
gen.eval()

# ── 1. Black-Scholes on GBM paths ────────────────────────────────────────────
print("Computing Black-Scholes baseline...")
prices_gbm_np = generate_gbm_paths(n_paths=N_PATHS, n_steps=N_STEPS, seed=1234)
bs_hedges     = bs_hedge_paths_fast(prices_gbm_np, K=STRIKE, T=1.0, sigma=0.2)
pnl_bs        = compute_pnl(prices_gbm_np, bs_hedges, strike=STRIKE, transaction_cost=TC)
cvar_bs       = compute_cvar(pnl_bs, alpha=0.1)
print(f"  BS CVaR: {cvar_bs:.4f}")
torch.cuda.empty_cache()
# ── 2. Deep Hedger on GBM paths ───────────────────────────────────────────────
print("Computing Deep Hedger on GBM paths...")
with torch.no_grad():
    prices_gbm_t   = torch.tensor(prices_gbm_np, dtype=torch.float32).to(DEVICE)
    features_gbm_t = prepare_features(prices_gbm_np, n_steps=N_STEPS).to(DEVICE)
    hedges_dh      = hedger(features_gbm_t)
    pnl_dh         = compute_pnl_torch(prices_gbm_t, hedges_dh, strike=STRIKE, tc=TC)
    pnl_dh_np      = pnl_dh.cpu().numpy()
cvar_dh = compute_cvar(pnl_dh_np, alpha=0.1)
print(f"  Deep Hedger CVaR (GBM): {cvar_dh:.4f}")
torch.cuda.empty_cache()
# ── 3. Adversarial Hedger on adversarial paths ────────────────────────────────
print("Computing Adversarial Hedger on adversarial paths...")
with torch.no_grad():
    z            = sample_noise(N_PATHS, N_STEPS, NOISE_SIZE, DEVICE)
    log_returns  = gen(z)
    log_returns  = constrain_log_returns(log_returns, scale=2.5)
    prices_adv_t = generate_paths_from_returns(log_returns, S0=STRIKE)
    features_adv = prepare_features(prices_adv_t.cpu().numpy(), n_steps=N_STEPS).to(DEVICE)
    hedges_adv   = hedger(features_adv)
    pnl_adv      = compute_pnl_torch(prices_adv_t, hedges_adv, strike=STRIKE, tc=TC)
    pnl_adv_np   = pnl_adv.cpu().numpy()
cvar_adv = compute_cvar(pnl_adv_np, alpha=0.1)
print(f"  Adversarial CVaR: {cvar_adv:.4f}")

# ── Print Summary Table ───────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"{'Method':<35} {'CVaR':>8}")
print("="*50)
print(f"{'Black-Scholes (GBM paths)':<35} {cvar_bs:>8.4f}")
print(f"{'Deep Hedger (GBM paths)':<35} {cvar_dh:>8.4f}")
print(f"{'Deep Hedger (adversarial paths)':<35} {cvar_adv:>8.4f}")
print("="*50)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot 1: P&L distributions ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.hist(pnl_bs,     bins=120, alpha=0.5, density=True,
         color="steelblue", label=f"Black-Scholes  (CVaR={cvar_bs:.2f})")
ax1.hist(pnl_dh_np,  bins=120, alpha=0.5, density=True,
         color="seagreen",  label=f"Deep Hedger GBM  (CVaR={cvar_dh:.2f})")
ax1.hist(pnl_adv_np, bins=120, alpha=0.5, density=True,
         color="tomato",    label=f"Adversarial Paths  (CVaR={cvar_adv:.2f})")
ax1.axvline(np.percentile(pnl_bs,     10), color="steelblue",
            linestyle="--", linewidth=1.5)
ax1.axvline(np.percentile(pnl_dh_np,  10), color="seagreen",
            linestyle="--", linewidth=1.5)
ax1.axvline(np.percentile(pnl_adv_np, 10), color="tomato",
            linestyle="--", linewidth=1.5)
ax1.set_title("P&L Distribution Comparison (dashed lines = CVaR cutoff at 10%)",
              fontsize=13)
ax1.set_xlabel("P&L")
ax1.set_ylabel("Density")
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# ── Plot 2: CVaR bar chart ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
methods = ["Black-Scholes\n(GBM)", "Deep Hedger\n(GBM)", "Deep Hedger\n(Adversarial)"]
cvars   = [cvar_bs, cvar_dh, cvar_adv]
colors  = ["steelblue", "seagreen", "tomato"]
bars    = ax2.bar(methods, cvars, color=colors, alpha=0.8, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, cvars):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_title("CVaR Comparison (lower = better)", fontsize=12)
ax2.set_ylabel("CVaR (10%)")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, max(cvars) * 1.2)

# ── Plot 3: Sample adversarial vs GBM paths ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
n_show = 30
ax3.plot(prices_gbm_np[:n_show].T,
         alpha=0.3, linewidth=0.8, color="steelblue")
ax3.plot(prices_adv_t.cpu().numpy()[:n_show].T,
         alpha=0.3, linewidth=0.8, color="tomato")
ax3.axhline(STRIKE, color="black", linestyle="--",
            linewidth=1.0, label="Strike (K=100)")
# Legend proxies
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color="steelblue", linewidth=1.5, label="GBM paths"),
    Line2D([0], [0], color="tomato",    linewidth=1.5, label="Adversarial paths"),
    Line2D([0], [0], color="black",     linewidth=1.0,
           linestyle="--", label="Strike K=100"),
]
ax3.legend(handles=handles, fontsize=9)
ax3.set_title("Sample Price Paths: GBM vs Adversarial", fontsize=12)
ax3.set_xlabel("Time Step")
ax3.set_ylabel("Stock Price")
ax3.grid(alpha=0.3)

plt.suptitle("Adversarial Deep Hedging — Final Results", fontsize=15, fontweight="bold")
plt.savefig("notebooks/final_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFinal results plot saved to notebooks/final_results.png")