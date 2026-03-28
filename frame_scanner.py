from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np



@dataclass
class CameraModel:
    K: np.ndarray       # (3, 3) intrinsic matrix
    dist: np.ndarray    # (5,)   distortion coefficients



@dataclass
class BoardPose:
    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    success: bool = False



@dataclass
class CanvasSpec:
    width_mm: float
    height_mm: float
    spacing_mm: float



CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_MM = 15.4    # ✅ UPDATED: physical square side length (mm)
CHARUCO_MARKER_MM = 11.5    # ✅ FIXED: must be < CHARUCO_SQUARE_MM (~75% = 11.55 mm)


BACKGROUND_THRESHOLD = 240  # pixels above this on all channels are skipped



def init_camera() -> Dict: #starts the camera and gets params
    import pyzed.sl as sl


    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA


    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED failed to open: {status}")


    calib = zed.get_camera_information().calibration_parameters.left_cam
    K = np.array([
        [calib.fx,       0, calib.cx],
        [       0, calib.fy, calib.cy],
        [       0,        0,        1],
    ], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)


    return {"zed_handle": zed, "camera_model": CameraModel(K=K, dist=dist)}



def detect_frame(zed_handle,
                 camera_model: CameraModel,
                 frame_bgr: Optional[np.ndarray] = None) -> BoardPose:
    import cv2
    import cv2.aruco as aruco


    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_MM,
        CHARUCO_MARKER_MM,
        aruco_dict,
    )


    if frame_bgr is None:
        import pyzed.sl as sl
        image_zed = sl.Mat()
        zed_handle.retrieve_image(image_zed, sl.VIEW.LEFT)
        frame_bgr = image_zed.get_data()[:, :, :3]
 
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) #grayscale for easier & better detection


    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters()) #1st checks for aruco only
    corners, ids, _ = detector.detectMarkers(gray)


    if ids is None or len(ids) == 0: #stops if didn`t find any
        return BoardPose(success=False)


    charuco_detector = aruco.CharucoDetector(board) #2nd iteration, checks for charuco based on previous aruco detection
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)


    if charuco_corners is None or len(charuco_corners) < 4: #at least 4 corners to solve pose
        return BoardPose(success=False)


    success, rvec, tvec = aruco.estimatePoseCharucoBoard( # PnP
        charuco_corners, charuco_ids, board,
        camera_model.K, camera_model.dist, None, None,
    )


    if not success:
        return BoardPose(success=False)


    return BoardPose(rvec=rvec.flatten(), tvec=tvec.flatten(), success=True)



def build_grid(spec: CanvasSpec) -> np.ndarray:
    nx = int(spec.width_mm  // spec.spacing_mm)
    ny = int(spec.height_mm // spec.spacing_mm)


    xs = np.arange(nx, dtype=np.float32) * spec.spacing_mm
    ys = np.arange(ny, dtype=np.float32) * spec.spacing_mm


    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])



def map_image(points_2d: np.ndarray,
              spec: CanvasSpec,
              image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image


    nx = int(spec.width_mm  // spec.spacing_mm)
    ny = int(spec.height_mm // spec.spacing_mm)


    img = Image.open(image_path).convert("RGB")
    img = img.resize((nx, ny), Image.LANCZOS)


    colors_flat = np.array(img, dtype=np.uint8).reshape(-1, 3)
    mask = ~np.all(colors_flat > BACKGROUND_THRESHOLD, axis=1)


    return points_2d[mask], colors_flat[mask]



def canvas_to_3d(points_2d: np.ndarray, pose: BoardPose) -> np.ndarray: #transforms 2d points to 3d (based on pose of the board)
    import cv2


    R, _ = cv2.Rodrigues(pose.rvec)
    pts = np.hstack([points_2d, np.zeros((len(points_2d), 1), dtype=np.float32)])
    return ((R @ pts.T).T + pose.tvec).astype(np.float32)



def export_points(points_3d: np.ndarray,
                  colors_rgb: np.ndarray,
                  output_path: str = "tuft_points.json") -> None:
    records = [
        {
            "x": float(round(pt[0], 3)),
            "y": float(round(pt[1], 3)),
            "z": float(round(pt[2], 3)),
            "r": int(col[0]),
            "g": int(col[1]),
            "b": int(col[2]),
        }
        for pt, col in zip(points_3d, colors_rgb)
    ]
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)