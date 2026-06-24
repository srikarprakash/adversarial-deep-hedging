# train_adversarial.py
# Fixed adversarial training loop with:
# 1. Gradient penalty (forces Generator diversity)
# 2. Noise scheduling (Generator explores more early on)
# 3. 5:1 Hedger:Generator update ratio (prevents Generator from overpowering)

import torch
import numpy as np
import matplotlib.pyplot as plt

from hedger import Hedger, prepare_features, cvar_loss, compute_pnl_torch
from generator import (Generator, generate_paths_from_returns,
                       sample_noise, constrain_log_returns)

# ── Config ────────────────────────────────────────────────────────────────────
N_STEPS         = 30
NOISE_SIZE      = 8
HIDDEN_SIZE     = 64
N_LAYERS        = 2
BATCH_SIZE      = 2048
N_EPOCHS        = 400
LR_HEDGER       = 1e-3
LR_GEN          = 5e-4      # slower generator learning rate
N_HEDGER_STEPS  = 5         # 5 hedger updates per 1 generator update
GP_WEIGHT       = 10.0      # gradient penalty strength
STRIKE          = 100.0
TC              = 0.001
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")
print(f"Epochs: {N_EPOCHS} | Batch: {BATCH_SIZE} | Hedger steps per Gen step: {N_HEDGER_STEPS}\n")

# ── Models ────────────────────────────────────────────────────────────────────
hedger = Hedger(input_size=2, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS).to(DEVICE)
gen    = Generator(noise_size=NOISE_SIZE, hidden_size=HIDDEN_SIZE,
                   n_layers=N_LAYERS, n_steps=N_STEPS).to(DEVICE)

# Warm start hedger from pretrained weights
hedger.load_state_dict(torch.load("models/hedger.pth", map_location=DEVICE))
print("Loaded pretrained hedger weights.")

opt_hedger = torch.optim.Adam(hedger.parameters(), lr=LR_HEDGER)
opt_gen    = torch.optim.Adam(gen.parameters(),    lr=LR_GEN)

sched_hedger = torch.optim.lr_scheduler.CosineAnnealingLR(opt_hedger, T_max=N_EPOCHS)
sched_gen    = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen,    T_max=N_EPOCHS)

# ── Noise schedule: start loose (explore), tighten over time ─────────────────
def get_noise_scale(epoch, n_epochs, start=5.0, end=2.5):
    """Linearly decay the constraint from 5x to 2.5x sigma*sqrt(dt)"""
    return start - (start - end) * (epoch / n_epochs)

# ── Gradient Penalty ──────────────────────────────────────────────────────────
def gradient_penalty(gen, z1, z2):
    """
    Penalizes Generator for producing similar outputs from different noise.
    Forces diversity in generated paths.
    CuDNN disabled for double backward compatibility.
    """
    alpha  = torch.rand(z1.shape[0], 1, 1, device=DEVICE)
    z_mix  = (alpha * z1 + (1 - alpha) * z2).requires_grad_(True)

    # Disable CuDNN for double backward pass through LSTM
    with torch.backends.cudnn.flags(enabled=False):
        log_ret_mix = gen(z_mix)

    grad = torch.autograd.grad(
        outputs=log_ret_mix.sum(),
        inputs=z_mix,
        create_graph=True
    )[0]

    gp = ((grad.norm(2, dim=-1) - 1) ** 2).mean()
    return gp
# ── Helpers ───────────────────────────────────────────────────────────────────
def gen_batch(batch_size, noise_scale):
    z           = sample_noise(batch_size, N_STEPS, NOISE_SIZE, DEVICE)
    log_returns = gen(z)
    log_returns = constrain_log_returns(log_returns, scale=noise_scale)
    prices      = generate_paths_from_returns(log_returns, S0=STRIKE)
    return prices, z

def prices_to_features(prices_t):
    prices_np = prices_t.detach().cpu().numpy()
    return prepare_features(prices_np, n_steps=N_STEPS).to(DEVICE)

# ── Training Loop ─────────────────────────────────────────────────────────────
hedger_losses = []
gen_losses    = []
gp_losses     = []

for epoch in range(1, N_EPOCHS + 1):

    hedger.train()
    gen.train()

    noise_scale = get_noise_scale(epoch, N_EPOCHS)

    # ── Phase 1: Train Hedger N_HEDGER_STEPS times ────────────────────────
    h_loss_epoch = []
    for _ in range(N_HEDGER_STEPS):
        with torch.no_grad():
            prices_t, _ = gen_batch(BATCH_SIZE, noise_scale)

        features_t = prices_to_features(prices_t)

        opt_hedger.zero_grad()
        hedges = hedger(features_t)
        pnl    = compute_pnl_torch(prices_t, hedges, strike=STRIKE, tc=TC)
        h_loss = cvar_loss(pnl, alpha=0.1)
        h_loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger.parameters(), max_norm=1.0)
        opt_hedger.step()
        h_loss_epoch.append(h_loss.item())

    # ── Phase 2: Train Generator once ────────────────────────────────────
    opt_gen.zero_grad()

    # Two different noise samples for gradient penalty
    prices_t, z1 = gen_batch(BATCH_SIZE, noise_scale)
    _,         z2 = gen_batch(BATCH_SIZE, noise_scale)

    features_t = prices_to_features(prices_t)

    with torch.no_grad():
        hedges = hedger(features_t)

    pnl    = compute_pnl_torch(prices_t, hedges.detach(), strike=STRIKE, tc=TC)
    g_loss = -cvar_loss(pnl, alpha=0.1)   # maximize CVaR

    # Add gradient penalty to discourage mode collapse
    gp     = gradient_penalty(gen, z1.detach(), z2.detach())
    total_gen_loss = g_loss + GP_WEIGHT * gp

    total_gen_loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
    opt_gen.step()

    sched_hedger.step()
    sched_gen.step()

    hedger_losses.append(np.mean(h_loss_epoch))
    gen_losses.append(-g_loss.item())
    gp_losses.append(gp.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}/{N_EPOCHS} | "
              f"Hedger CVaR: {np.mean(h_loss_epoch):.4f} | "
              f"Generator CVaR: {-g_loss.item():.4f} | "
              f"GP: {gp.item():.4f} | "
              f"Noise scale: {noise_scale:.2f}")

# ── Save Models ───────────────────────────────────────────────────────────────
torch.save(hedger.state_dict(), "models/adversarial_hedger.pth")
torch.save(gen.state_dict(),    "models/adversarial_generator.pth")
print("\nModels saved.")

# ── Plot Training Curves ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

ax1.plot(hedger_losses, label="Hedger CVaR (minimizing)",
         color="steelblue", linewidth=1.2)
ax1.plot(gen_losses,    label="Generator CVaR (maximizing)",
         color="tomato", linewidth=1.2)
ax1.set_title("Adversarial Training: Min-Max Game")
ax1.set_ylabel("CVaR")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(gp_losses, color="green", linewidth=1.0)
ax2.set_title("Gradient Penalty (should stabilize near 0)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("GP Loss")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/adversarial_training_curve.png", dpi=150)
plt.show()
print("Training curve saved.")

# ── Final Evaluation ──────────────────────────────────────────────────────────
print("\n── Final Evaluation ──")
hedger.eval()
gen.eval()

with torch.no_grad():
    # Adversarial paths
    prices_adv, _ = gen_batch(50000, noise_scale=2.5)
    features_adv  = prices_to_features(prices_adv)
    hedges_adv    = hedger(features_adv)
    pnl_adv       = compute_pnl_torch(prices_adv, hedges_adv,
                                       strike=STRIKE, tc=TC)
    pnl_adv_np    = pnl_adv.cpu().numpy()

    # GBM paths
    from gbm import generate_gbm_paths
    prices_gbm_np  = generate_gbm_paths(n_paths=50000, n_steps=N_STEPS, seed=1234)
    prices_gbm_t   = torch.tensor(prices_gbm_np, dtype=torch.float32).to(DEVICE)
    features_gbm_t = prepare_features(prices_gbm_np, n_steps=N_STEPS).to(DEVICE)
    hedges_gbm     = hedger(features_gbm_t)
    pnl_gbm        = compute_pnl_torch(prices_gbm_t, hedges_gbm,
                                        strike=STRIKE, tc=TC)
    pnl_gbm_np     = pnl_gbm.cpu().numpy()

cvar_adv = -np.percentile(pnl_adv_np, 10)
cvar_gbm = -np.percentile(pnl_gbm_np, 10)

print(f"CVaR on adversarial paths: {cvar_adv:.4f}")
print(f"CVaR on GBM paths:         {cvar_gbm:.4f}")
print(f"P&L mean (adversarial):    {pnl_adv_np.mean():.4f}")
print(f"P&L std  (adversarial):    {pnl_adv_np.std():.4f}")

# ── Comparison Plot ───────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.hist(pnl_gbm_np, bins=100, alpha=0.5, color="steelblue", density=True,
         label=f"GBM paths (CVaR={cvar_gbm:.2f})")
plt.hist(pnl_adv_np, bins=100, alpha=0.5, color="tomato", density=True,
         label=f"Adversarial paths (CVaR={cvar_adv:.2f})")
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
print("Comparison plot saved.")