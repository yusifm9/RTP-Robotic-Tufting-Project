import numpy as np
import matplotlib.pyplot as plt

csv_path = "canvas_state_001_uvz.csv"  # adjust if needed

# 1. Load uvz data
data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
u = data[:, 0]      # u_mm
v = data[:, 1]      # v_mm
z = data[:, 2]      # z_mm_board (almost 0 in your current planar export)

# ─────────────────────────────────────────────────────────────
# 2. (Optional) add synthetic sag to simulate a deformable canvas
#    Set sag_amplitude_mm = 0.0 if you want to fit the original planar data.
sag_amplitude_mm = 5.0   # try 0.0, 3.0, 5.0, 10.0
sigma_mm         = 100.0 # spatial spread of sag (higher = gentler sag)

u_center = 0.5 * (u.max() + u.min())
v_center = 0.5 * (v.max() + v.min())

sag = sag_amplitude_mm * np.exp(
    -(((u - u_center) ** 2 + (v - v_center) ** 2) / (2.0 * sigma_mm ** 2))
)

z_effective = z + sag
# ─────────────────────────────────────────────────────────────

# 3. Fit plane: z = a*u + b*v + c to z_effective
A = np.column_stack([u, v, np.ones_like(u)])
coeffs, *_ = np.linalg.lstsq(A, z_effective, rcond=None)
a, b, c = coeffs

print("Plane coefficients: z = a*u + b*v + c")
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")

z_pred = A @ coeffs
rmse = np.sqrt(np.mean((z_effective - z_pred) ** 2))
print(f"RMSE (mm) with sag_amplitude_mm={sag_amplitude_mm}: {rmse}")

# 4. Residual heatmap
residuals = z_effective - z_pred

plt.figure(figsize=(6, 5))
sc = plt.scatter(u, v, c=residuals, cmap="coolwarm", s=20)
plt.colorbar(sc, label="residual (mm)")
plt.xlabel("u_mm")
plt.ylabel("v_mm")
plt.title(f"Planar baseline residuals (sag={sag_amplitude_mm} mm)")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()