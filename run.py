import cv2
import numpy as np
from frame_scanner import init_camera, detect_frame, build_grid, CANVAS_SPEC

# 1. Start camera
cam = init_camera()
zed = cam["zed_handle"]
camera_model = cam["camera_model"]

# 2. Run live detection loop
print("Pointing at ChArUco board... press Q to quit.")
while True:
    pose = detect_frame(zed, camera_model)

    if pose.success:
        print(f"✅ Board detected! tvec = {pose.tvec.round(1)} mm")

        # 3. Project canvas corners onto the live image
        import pyzed.sl as sl
        image_zed = sl.Mat()
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        frame = image_zed.get_data()[:, :, :3].copy()

        canvas_corners_3d = np.array([
            [0,           0,           0],   # top-left (board origin)
            [CANVAS_SPEC.width_mm,  0,           0],   # top-right
            [CANVAS_SPEC.width_mm,  CANVAS_SPEC.height_mm, 0],   # bottom-right
            [0,           CANVAS_SPEC.height_mm, 0],   # bottom-left
        ], dtype=np.float32)

        projected, _ = cv2.projectPoints(
            canvas_corners_3d,
            pose.rvec, pose.tvec,
            camera_model.K, camera_model.dist
        )
        projected = projected.reshape(-1, 2).astype(int)

        # Draw rectangle on frame
        for i in range(4):
            cv2.line(frame, projected[i], projected[(i+1) % 4], (0, 255, 0), 2)

        cv2.imshow("Canvas projection", frame)
    else:
        print("❌ Board not detected — check lighting and angle.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.close()
cv2.destroyAllWindows()