from __future__ import annotations

import argparse
import csv
import time

from robot_config import (
    MM_TO_M,
    ROBOT_BASE_ORIGIN_M,
    SAFE_Z_OFFSET_M,
    CONTACT_Z_OFFSET_M,
    DEFAULT_TCP_QUAT,
    MAX_VEL,
    DWELL_TIME_S,
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


def pose_to_movel_primitive(pose: list[float]) -> str:
    pose_text = ", ".join(str(v) for v in pose)

    return (
        f"MoveL(target=[{pose_text}], "
        f"maxVel={MAX_VEL})"
    )


def simulate(csv_path: str, max_points: int, sleep: bool = False) -> None:
    points = load_points_csv(csv_path)
    points = points[:max_points]

    print("\nOFFLINE FLEXIV MOTION SIMULATION")
    print("--------------------------------")
    print(f"CSV file: {csv_path}")
    print(f"Loaded points: {len(points)}")
    print(f"Robot base origin: {ROBOT_BASE_ORIGIN_M}")
    print(f"Safe Z offset: {SAFE_Z_OFFSET_M} m")
    print(f"Contact Z offset: {CONTACT_Z_OFFSET_M} m")
    print(f"Max velocity: {MAX_VEL}")
    print(f"Dwell time: {DWELL_TIME_S} s")

    for i, point in enumerate(points):
        safe_pose = canvas_point_to_robot_pose(point, contact=False)
        contact_pose = canvas_point_to_robot_pose(point, contact=True)

        safe_cmd = pose_to_movel_primitive(safe_pose)
        contact_cmd = pose_to_movel_primitive(contact_pose)

        print(f"\nPoint {i + 1}/{len(points)}")
        print(f"Input CSV point mm: {point}")

        print("  Step 1: move above point")
        print(f"    {safe_cmd}")

        print("  Step 2: move down/contact")
        print(f"    {contact_cmd}")

        print(f"  Step 3: dwell for {DWELL_TIME_S} s")

        print("  Step 4: move back above point")
        print(f"    {safe_cmd}")

        if sleep:
            time.sleep(0.2)

    print("\nSimulation finished. No robot was connected or moved.")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file with X,Y,Z columns",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=5,
        help="Maximum number of points to simulate",
    )

    parser.add_argument(
        "--sleep",
        action="store_true",
        help="Slow down printed simulation",
    )

    args = parser.parse_args()

    simulate(
        csv_path=args.csv_path,
        max_points=args.max_points,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()