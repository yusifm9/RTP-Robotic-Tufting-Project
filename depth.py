import pyzed.sl as sl
import numpy as np

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.MILLIMETER
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
zed.open(init_params)

runtime_params = sl.RuntimeParameters()
point_cloud = sl.Mat()

# Grab a few frames first to let the camera warm up
for _ in range(10):
    zed.grab(runtime_params)

zed.grab(runtime_params)  # grab a fresh frame
zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)  # then retrieve depth

h, w = point_cloud.get_height(), point_cloud.get_width()
test_points = [
    (w//2,   h//2),
    (w//4,   h//4),
    (3*w//4, h//4),
    (w//4,   3*h//4),
    (3*w//4, 3*h//4),
]

print("Sampling depth at 5 points across the canvas:")
for (x, y) in test_points:
    err, point = point_cloud.get_value(x, y)
    z = point[2]
    if np.isfinite(z):
        print(f"  Pixel ({x:4d}, {y:4d}) → Z = {z:.1f} mm ✅")
    else:
        print(f"  Pixel ({x:4d}, {y:4d}) → Z = NaN ❌")

zed.close()