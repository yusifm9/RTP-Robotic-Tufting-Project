import cv2
import cv2.aruco as aruco
import pyzed.sl as sl
import numpy as np

# Try ALL common dictionaries to find which one matches your board
DICTS_TO_TRY = {
    "DICT_4X4_50":   aruco.DICT_4X4_50,
    "DICT_4X4_100":  aruco.DICT_4X4_100,
    "DICT_5X5_50":   aruco.DICT_5X5_50,
    "DICT_5X5_100":  aruco.DICT_5X5_100,
    "DICT_6X6_50":   aruco.DICT_6X6_50,
    "DICT_6X6_250":  aruco.DICT_6X6_250,
}

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.MILLIMETER
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
zed.open(init_params)

image_zed = sl.Mat()
print("Running dictionary scan... press Q to quit.")

while True:
    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    frame = image_zed.get_data()[:, :, :3].copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found_any = False
    for name, dict_id in DICTS_TO_TRY.items():
        d = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(d, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            print(f"✅ Found {len(ids)} markers with {name}!")
            aruco.drawDetectedMarkers(frame, corners, ids)
            found_any = True

    if not found_any:
        cv2.putText(frame, "No markers found in ANY dictionary", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Dictionary Scanner", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.close()
cv2.destroyAllWindows()