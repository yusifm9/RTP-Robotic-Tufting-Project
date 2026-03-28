import cv2
import numpy as np
from frame_scanner import init_camera, detect_frame, build_grid, CANVAS_SPEC

# 1. Start camera
cam = init_camera()
zed = cam["zed_handle"]
camera_model = cam["camera_model"]

print("Pointing at ChArUco board... press Q to quit.")

import pyzed.sl as sl
image_zed = sl.Mat()

while True:
    # Always grab and show the live frame first
    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    frame = image_zed.get_data()[:, :, :3].copy()

    pose = detect_frame(zed, camera_model, frame_bgr=frame)  # pass frame so it doesn't grab twice

    if pose.success:
        print(f"✅ Board detected! tvec = {pose.tvec.round(1)} mm")

        canvas_corners_3d = np.array([
            [0,                    0,                     0],
            [CANVAS_SPEC.width_mm, 0,                     0],
            [CANVAS_SPEC.width_mm, CANVAS_SPEC.height_mm, 0],
            [0,                    CANVAS_SPEC.height_mm, 0],
        ], dtype=np.float32)

        projected, _ = cv2.projectPoints(
            canvas_corners_3d,
            pose.rvec, pose.tvec,
            camera_model.K, camera_model.dist
        )
        projected = projected.reshape(-1, 2).astype(int)

        for i in range(4):
            cv2.line(frame, projected[i], projected[(i+1) % 4], (0, 255, 0), 2)

        cv2.putText(frame, "Board detected!", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No board detected - adjust angle/lighting", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("ZED Camera - ChArUco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.close()
cv2.destroyAllWindows()