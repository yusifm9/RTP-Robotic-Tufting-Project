import numpy as np
import matplotlib.pyplot as plt

csv_path = "canvas_state_001_uvz.csv"

# 1. Load data
data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
u = data[:, 0]
v = data[:, 1]
z = data[:, 2]

# 2. Add the same synthetic sag as before
sag_amplitude_mm = 5.0
sigma_mm         = 100.0

u_center = 0.5 * (u.max() + u.min())
v_center = 0.5 * (v.max() + v.min())

sag = sag_amplitude_mm * np.exp(
    -(((u - u_center) ** 2 + (v - v_center) ** 2) / (2.0 * sigma_mm ** 2))
)
z_effective = z + sag

# 3. Build polynomial feature matrix (degree 2)
# Features: [u^2, v^2, u*v, u, v, 1]
A = np.column_stack([
    u**2,
    v**2,
    u * v,
    u,
    v,
    np.ones_like(u),
])

# 4. Fit polynomial surface (least squares)
coeffs, *_ = np.linalg.lstsq(A, z_effective, rcond=None)
a2, b2, c2, d2, e2, f2 = coeffs

print("Polynomial (degree-2) coefficients:")
print(f"  u^2 : {a2}")
print(f"  v^2 : {b2}")
print(f"  u*v : {c2}")
print(f"  u   : {d2}")
print(f"  v   : {e2}")
print(f"  1   : {f2}")

z_pred = A @ coeffs
residuals = z_effective - z_pred

rmse_poly = np.sqrt(np.mean(residuals ** 2))
print(f"\nRMSE polynomial (mm): {rmse_poly}")

# 5. Planar baseline RMSE for direct comparison
A_plane = np.column_stack([u, v, np.ones_like(u)])
coeffs_plane, *_ = np.linalg.lstsq(A_plane, z_effective, rcond=None)
z_pred_plane = A_plane @ coeffs_plane
rmse_plane = np.sqrt(np.mean((z_effective - z_pred_plane) ** 2))
print(f"RMSE planar    (mm): {rmse_plane}")
print(f"\nImprovement: {(1 - rmse_poly / rmse_plane) * 100:.1f}% lower RMSE with polynomial")

# 6. Side-by-side residual heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

vmax = max(
    np.abs(z_effective - z_pred_plane).max(),
    np.abs(residuals).max()
)

for ax, res, title in zip(
    axes,
    [z_effective - z_pred_plane, residuals],
    ["Planar baseline", "Polynomial (degree-2) baseline"],
):
    sc = ax.scatter(u, v, c=res, cmap="coolwarm", s=20,
                    vmin=-vmax, vmax=vmax)
    fig.colorbar(sc, ax=ax, label="residual (mm)")
    ax.set_xlabel("u_mm")
    ax.set_ylabel("v_mm")
    ax.set_title(f"{title}\nRMSE = {np.sqrt(np.mean(res**2)):.4f} mm")
    ax.set_aspect("equal")

plt.suptitle(f"Synthetic sag = {sag_amplitude_mm} mm", fontsize=13)
plt.tight_layout()
plt.show()