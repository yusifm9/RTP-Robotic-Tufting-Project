from __future__ import annotations

import csv
from pathlib import Path

from robot_config import (
    MM_TO_M,
    ROBOT_BASE_ORIGIN_M,
    SAFE_Z_OFFSET_M,
    CONTACT_Z_OFFSET_M,
    DEFAULT_TCP_QUAT,
)


def load_points_csv(csv_path: str) -> list[dict[str, float]]:
    points = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        required = {"X", "Y", "Z"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {required}")

        for row in reader:
            points.append({
                "x_mm": float(row["X"]),
                "y_mm": float(row["Y"]),
                "z_mm": float(row["Z"]),
            })

    return points


def canvas_point_to_robot_pose(point: dict[str, float], contact: bool) -> list[float]:
    """
    Convert canvas point to robot pose.

    Output:
        [x, y, z, qw, qx, qy, qz]

    This is a placeholder transform.
    Later, replace it with real camera/canvas-to-robot calibration.
    """

    x_m = ROBOT_BASE_ORIGIN_M["x"] + point["x_mm"] * MM_TO_M
    y_m = ROBOT_BASE_ORIGIN_M["y"] + point["y_mm"] * MM_TO_M

    if contact:
        z_m = ROBOT_BASE_ORIGIN_M["z"] + CONTACT_Z_OFFSET_M
    else:
        z_m = ROBOT_BASE_ORIGIN_M["z"] + SAFE_Z_OFFSET_M

    return [
        round(x_m, 6),
        round(y_m, 6),
        round(z_m, 6),
        *DEFAULT_TCP_QUAT,
    ]


def main() -> None:
    csv_path = "robot_control/test_points.csv"

    points = load_points_csv(csv_path)

    print(f"Loaded {len(points)} points from {csv_path}")

    for i, point in enumerate(points):
        safe_pose = canvas_point_to_robot_pose(point, contact=False)
        contact_pose = canvas_point_to_robot_pose(point, contact=True)

        print(f"\nPoint {i + 1}")
        print(f"CSV point:     {point}")
        print(f"Safe pose:     {safe_pose}")
        print(f"Contact pose:  {contact_pose}")


if __name__ == "__main__":
    main()