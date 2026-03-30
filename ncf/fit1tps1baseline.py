import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

csv_path = "canvas_state_001_uvz.csv"

# 1. Load data
data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
u = data[:, 0]
v = data[:, 1]
z = data[:, 2]

# 2. Same synthetic sag for fair comparison
sag_amplitude_mm = 5.0
sigma_mm         = 100.0

u_center = 0.5 * (u.max() + u.min())
v_center = 0.5 * (v.max() + v.min())

sag = sag_amplitude_mm * np.exp(
    -(((u - u_center) ** 2 + (v - v_center) ** 2) / (2.0 * sigma_mm ** 2))
)
z_effective = z + sag

# 3. Split into train (80%) and test (20%) for leave-out evaluation
np.random.seed(42)
n = len(u)
idx = np.random.permutation(n)
train_idx = idx[:int(0.8 * n)]
test_idx  = idx[int(0.8 * n):]

uv_train = np.column_stack([u[train_idx], v[train_idx]])
uv_test  = np.column_stack([u[test_idx],  v[test_idx]])
z_train  = z_effective[train_idx]
z_test   = z_effective[test_idx]

# 4. Fit TPS (thin-plate spline) on training points
tps = RBFInterpolator(uv_train, z_train, kernel="thin_plate_spline")

# 5. Predict on ALL points for heatmap
uv_all = np.column_stack([u, v])
z_pred_tps = tps(uv_all)
residuals_tps = z_effective - z_pred_tps
rmse_tps = np.sqrt(np.mean(residuals_tps ** 2))

# 6. Planar baseline for comparison
A_plane = np.column_stack([u, v, np.ones_like(u)])
coeffs_plane, *_ = np.linalg.lstsq(A_plane, z_effective, rcond=None)
z_pred_plane = A_plane @ coeffs_plane
rmse_plane = np.sqrt(np.mean((z_effective - z_pred_plane) ** 2))

# 7. Polynomial baseline for comparison
A_poly = np.column_stack([u**2, v**2, u*v, u, v, np.ones_like(u)])
coeffs_poly, *_ = np.linalg.lstsq(A_poly, z_effective, rcond=None)
z_pred_poly = A_poly @ coeffs_poly
rmse_poly = np.sqrt(np.mean((z_effective - z_pred_poly) ** 2))

# 8. Print comparison table
print("=" * 45)
print(f"{'Method':<25} {'RMSE (mm)':>10}")
print("=" * 45)
print(f"{'Planar (B1)':<25} {rmse_plane:>10.4f}")
print(f"{'Polynomial deg-2 (B3)':<25} {rmse_poly:>10.4f}")
print(f"{'Thin-plate spline (B4)':<25} {rmse_tps:>10.4f}")
print("=" * 45)

# 9. Three-way residual heatmap
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

vmax = max(
    np.abs(z_effective - z_pred_plane).max(),
    np.abs(z_effective - z_pred_poly).max(),
    np.abs(residuals_tps).max(),
)

for ax, res, title, rmse in zip(
    axes,
    [z_effective - z_pred_plane, z_effective - z_pred_poly, residuals_tps],
    ["Planar (B1)", "Polynomial deg-2 (B3)", "Thin-plate spline (B4)"],
    [rmse_plane, rmse_poly, rmse_tps],
):
    sc = ax.scatter(u, v, c=res, cmap="coolwarm", s=20, vmin=-vmax, vmax=vmax)
    fig.colorbar(sc, ax=ax, label="residual (mm)")
    ax.set_xlabel("u_mm")
    ax.set_ylabel("v_mm")
    ax.set_title(f"{title}\nRMSE = {rmse:.4f} mm")
    ax.set_aspect("equal")

plt.suptitle(f"Baseline comparison — synthetic sag = {sag_amplitude_mm} mm", fontsize=13)
plt.tight_layout()
plt.show()