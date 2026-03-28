import json
import numpy as np
import matplotlib.pyplot as plt

with open("tuft_points.json", "r") as f:
    data = json.load(f)

points = np.array([[p["x"], p["y"], p["z"]] for p in data])
colors = np.array([[p["r"], p["g"], p["b"]] for p in data], dtype=float)

# Normalize RGB to [0, 1] if values are in 0–255 range
if colors.max() > 1.0:
    colors /= 255.0

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    c=colors,
    s=1,
    depthshade=True,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Point Cloud")

plt.tight_layout()
plt.savefig("output.png", dpi=150)
plt.show()