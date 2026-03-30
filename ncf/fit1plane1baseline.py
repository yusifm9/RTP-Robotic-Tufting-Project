import numpy as np
import matplotlib.pyplot as plt

csv_path = "canvas_state_001_uvz.csv"  # adjust if needed

# 1. Load data
data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
u = data[:, 0]      # u_mm
v = data[:, 1]      # v_mm
z = data[:, 2]      # z_mm_board

# 2. Fit a plane: z = a*u + b*v + c  (least squares)
A = np.column_stack([u, v, np.ones_like(u)])
coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
a, b, c = coeffs
print("Plane coefficients: z = a*u + b*v + c")
print(f"a = {a}, b = {b}, c = {c}")

# 3. Predict and compute RMSE
z_pred = A @ coeffs
rmse = np.sqrt(np.mean((z - z_pred) ** 2))
print("RMSE (mm):", rmse)

# 4. Visualize residuals
plt.figure(figsize=(6, 5))
sc = plt.scatter(u, v, c=z - z_pred, cmap="coolwarm", s=10)
plt.colorbar(sc, label="residual (mm)")
plt.xlabel("u_mm")
plt.ylabel("v_mm")
plt.title("Planar baseline residuals on canvas_state_001")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()