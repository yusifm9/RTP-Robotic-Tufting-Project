import numpy as np
from frame_scanner import CanvasSpec, build_grid, canvas_to_3d, BoardPose, export_points

spec = CanvasSpec(width_mm=300.0, height_mm=200.0, spacing_mm=10.0)
grid = build_grid(spec)
print(f"Grid points: {len(grid)}")
print(f"First 5 points:\n{grid[:5]}")

fake_pose = BoardPose(
    rvec=np.zeros(3, dtype=np.float32),
    tvec=np.array([0, 0, 600], dtype=np.float32),
    success=True,
)

pts_3d = canvas_to_3d(grid, fake_pose)
print(f"\nFirst 5 points in 3D (mm):\n{pts_3d[:5]}")

fake_colors = np.tile([255, 0, 0], (len(pts_3d), 1)).astype(np.uint8)
export_points(pts_3d, fake_colors, output_path="tuft_points_test.json")
print("\nExported to tuft_points_test.json")