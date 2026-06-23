# train_hedger.py
# Trains the Hedger network to minimize CVaR of P&L

import torch
import numpy as np
import matplotlib.pyplot as plt
from gbm import generate_gbm_paths
from hedger import Hedger, prepare_features, cvar_loss, compute_pnl_torch

# ── Config ──────────────────────────────────────────────────────────────────
N_PATHS      = 10000   # paths per epoch
N_STEPS      = 30      # trading steps
N_EPOCHS     = 200     # training epochs
BATCH_SIZE   = 2048    # paths per gradient update
LR           = 1e-3    # learning rate
STRIKE       = 100.0
TC           = 0.001   # transaction cost 0.1%
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")

# ── Model + Optimizer ────────────────────────────────────────────────────────
model     = Hedger(input_size=2, hidden_size=64, n_layers=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# ── Training Loop ────────────────────────────────────────────────────────────
history = []  # track CVaR loss each epoch

for epoch in range(1, N_EPOCHS + 1):

    # Generate fresh GBM paths every epoch (so model doesn't memorize)
    prices_np = generate_gbm_paths(
        n_paths=N_PATHS, n_steps=N_STEPS, seed=epoch  # different seed each epoch
    )
    prices_tensor  = torch.tensor(prices_np, dtype=torch.float32).to(DEVICE)
    features_tensor = prepare_features(prices_np).to(DEVICE)

    # Mini-batch training
    epoch_losses = []
    indices = torch.randperm(N_PATHS)

    for start in range(0, N_PATHS, BATCH_SIZE):
        batch_idx     = indices[start : start + BATCH_SIZE]
        batch_prices  = prices_tensor[batch_idx]
        batch_features = features_tensor[batch_idx]

        optimizer.zero_grad()

        # Forward pass: get hedge ratios
        hedges = model(batch_features)

        # Compute P&L for this batch
        pnl = compute_pnl_torch(batch_prices, hedges, strike=STRIKE, tc=TC)

        # Loss = CVaR (average of worst 10% outcomes)
        loss = cvar_loss(pnl, alpha=0.1)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    scheduler.step()

    avg_loss = np.mean(epoch_losses)
    history.append(avg_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{N_EPOCHS} | CVaR Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

# ── Save Model ───────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "models/hedger.pth")
print("\nModel saved to models/hedger.pth")

# ── Plot Training Curve ──────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(history, linewidth=1.5, color="steelblue")
plt.title("Hedger Training: CVaR Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("CVaR Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/training_curve.png", dpi=150)
plt.show()
print("Training curve saved to notebooks/training_curve.png")

# ── Final Evaluation ─────────────────────────────────────────────────────────
print("\n── Final Evaluation ──")
model.eval()
with torch.no_grad():
    prices_np   = generate_gbm_paths(n_paths=50000, n_steps=N_STEPS, seed=9999)
    prices_t    = torch.tensor(prices_np, dtype=torch.float32).to(DEVICE)
    features_t  = prepare_features(prices_np).to(DEVICE)
    hedges_t    = model(features_t)
    pnl_t       = compute_pnl_torch(prices_t, hedges_t, strike=STRIKE, tc=TC)
    pnl_np      = pnl_t.cpu().numpy()

final_cvar = -np.percentile(pnl_np, 10)
print(f"Final CVaR  (trained, 50k paths): {final_cvar:.4f}")
print(f"Final P&L mean:                   {pnl_np.mean():.4f}")
print(f"Final P&L std:                    {pnl_np.std():.4f}")

# ── P&L Distribution Plot ────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.hist(pnl_np, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
plt.axvline(np.percentile(pnl_np, 10), color="red",
            linestyle="--", linewidth=1.5, label=f"CVaR cutoff (10%)")
plt.title("P&L Distribution — Trained Hedger")
plt.xlabel("P&L")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/pnl_distribution.png", dpi=150)
plt.show()
print("P&L distribution saved to notebooks/pnl_distribution.png")