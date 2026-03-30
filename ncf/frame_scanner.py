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


CHARUCO_SQUARES_X = 8
CHARUCO_SQUARES_Y = 6
CHARUCO_SQUARE_MM = 15.4    # physical square side length (mm)
CHARUCO_MARKER_MM = 11.5    # must be < CHARUCO_SQUARE_MM (~75% = 11.55 mm)

CANVAS_SPEC = CanvasSpec(
    width_mm   = 400.0,   # total canvas width (mm)
    height_mm  = 805.0,   # total canvas height (mm)
    spacing_mm = 15.0,    # distance between tuft points (mm)
)

BACKGROUND_THRESHOLD = 240  # pixels above this on all channels are skipped


def init_camera() -> Dict:
    import pyzed.sl as sl

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED failed to open: {status}")

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
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

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
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

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return BoardPose(success=False)

    charuco_detector = aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

    if charuco_corners is None or len(charuco_corners) < 4:
        return BoardPose(success=False)

    success, rvec, tvec = aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board,
        camera_model.K, camera_model.dist, None, None,
    )

    if not success:
        return BoardPose(success=False)

    return BoardPose(rvec=rvec.flatten(), tvec=tvec.flatten(), success=True)


def build_grid(spec: CanvasSpec = CANVAS_SPEC) -> np.ndarray:
    nx = int(spec.width_mm  // spec.spacing_mm)
    ny = int(spec.height_mm // spec.spacing_mm)

    xs = np.arange(nx, dtype=np.float32) * spec.spacing_mm
    ys = np.arange(ny, dtype=np.float32) * spec.spacing_mm

    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def map_image(points_2d: np.ndarray,
              image_path: str,
              spec: CanvasSpec = CANVAS_SPEC) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    nx = int(spec.width_mm  // spec.spacing_mm)
    ny = int(spec.height_mm // spec.spacing_mm)

    img = Image.open(image_path).convert("RGB")
    img = img.resize((nx, ny), Image.LANCZOS)

    colors_flat = np.array(img, dtype=np.uint8).reshape(-1, 3)
    mask = ~np.all(colors_flat > BACKGROUND_THRESHOLD, axis=1)

    return points_2d[mask], colors_flat[mask]


def canvas_to_3d(points_2d: np.ndarray, pose: BoardPose) -> np.ndarray:
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


def export_uvz(points_2d: np.ndarray,
               points_3d: np.ndarray,
               pose: BoardPose,
               output_path: str = "canvas_state_001_uvz.csv") -> None:
    """
    Save (u, v, z_board) triples in the ChArUco/board frame.

    points_2d: (N, 2) canvas coordinates in mm (board frame, from build_grid/map_image)
    points_3d: (N, 3) camera-frame 3D coordinates from canvas_to_3d
    pose:      BoardPose giving board pose in camera frame
    """
    import cv2
    import csv

    # Camera -> board: P_board = R.T @ (P_cam - t)
    R_cam_board, _ = cv2.Rodrigues(pose.rvec)
    R_board_cam = R_cam_board.T
    t = pose.tvec.reshape(3, 1)

    pts_cam = points_3d.T  # (3, N)
    pts_board = R_board_cam @ (pts_cam - t)  # (3, N)

    z_board = pts_board[2, :]  # height in board frame

    u = points_2d[:, 0]
    v = points_2d[:, 1]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["u_mm", "v_mm", "z_mm_board"])
        for ui, vi, zi in zip(u, v, z_board):
            writer.writerow([float(ui), float(vi), float(zi)])


def visualise_points(output_path: str = "tuft_points.json") -> None:
    import matplotlib.pyplot as plt

    with open(output_path) as f:
        records = json.load(f)

    xs = [r["x"] for r in records]
    ys = [r["y"] for r in records]
    cs = [(r["r"] / 255, r["g"] / 255, r["b"] / 255) for r in records]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.scatter(xs, ys, c=cs, s=4, linewidths=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Tuft point map  ({len(records)} points)")
    plt.tight_layout()
    plt.show()


def run_pipeline(image_path: str,
                 output_path: str = "tuft_points.json",
                 uvz_path: str = "canvas_state_001_uvz.csv") -> None:
    import pyzed.sl as sl

    cam = init_camera()
    zed_handle   = cam["zed_handle"]
    camera_model = cam["camera_model"]

    runtime = sl.RuntimeParameters()
    if zed_handle.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        zed_handle.close()
        raise RuntimeError("ZED grab failed.")

    image_zed = sl.Mat()
    zed_handle.retrieve_image(image_zed, sl.VIEW.LEFT)
    frame_bgr = image_zed.get_data()[:, :, :3]

    pose = detect_frame(zed_handle, camera_model, frame_bgr=frame_bgr)
    if not pose.success:
        zed_handle.close()
        raise RuntimeError("Board not detected. Check lighting and board position.")

    points_2d         = build_grid()
    active_2d, colors = map_image(points_2d, image_path)
    points_3d         = canvas_to_3d(active_2d, pose)

    export_points(points_3d, colors, output_path)
    export_uvz(active_2d, points_3d, pose, uvz_path)
    visualise_points(output_path)

    zed_handle.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python frame_scanner.py <image.png>")
        sys.exit(1)

    run_pipeline(sys.argv[1])