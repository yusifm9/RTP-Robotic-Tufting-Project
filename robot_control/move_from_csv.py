from __future__ import annotations

import argparse
import flexivrdk
import csv
import time

from robot_config import (
    ROBOT_SN,
    MM_TO_M,
    ROBOT_BASE_ORIGIN_M,
    SAFE_Z_OFFSET_M,
    CONTACT_Z_OFFSET_M,
    DEFAULT_TCP_QUAT,
    MAX_VEL,
    DWELL_TIME_S,
)


def load_points_csv(csv_path: str) -> list[dict[str, float]]:
    """
    Load X, Y, Z points from CSV.

    Expected format:
        X,Y,Z
        0,0,0
        10,0,0
    """

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
    Convert one CSV point to one robot TCP pose.

    Output format:
        [x, y, z, qw, qx, qy, qz]

    WARNING:
    This is only a placeholder mapping.
    Replace this with real canvas-to-robot calibration later.
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


def dry_run(csv_path: str, max_points: int) -> None:
    points = load_points_csv(csv_path)
    points = points[:max_points]

    print(f"Loaded {len(points)} points from {csv_path}")
    print("No robot movement. Printing converted poses only.")

    for i, point in enumerate(points):
        safe_pose = canvas_point_to_robot_pose(point, contact=False)
        contact_pose = canvas_point_to_robot_pose(point, contact=True)

        print(f"\nPoint {i + 1}")
        print(f"CSV point:     {point}")
        print(f"Safe pose:     {safe_pose}")
        print(f"Contact pose:  {contact_pose}")


def wait_until_robot_done(robot, timeout_s: float = 30.0) -> None:
    """
    Wait until the robot finishes the current primitive.

    Flexiv examples use robot.busy() after ExecutePrimitive().
    """

    start_time = time.time()

    while robot.busy():
        if time.time() - start_time > timeout_s:
            raise TimeoutError("Robot primitive motion timed out.")

        time.sleep(0.1)


def pose_to_movel_primitive(pose: list[float]) -> str:
    """
    Build Flexiv MoveL primitive command string.

    Pose format:
        [x, y, z, qw, qx, qy, qz]
    """

    pose_text = ", ".join(str(v) for v in pose)

    return (
        f"MoveL(target=[{pose_text}], "
        f"maxVel={MAX_VEL})"
    )


def move_robot_from_csv(csv_path: str, max_points: int) -> None:
    """
    Move robot step-by-step through CSV points.

    Motion per point:
        1. Move above point
        2. Move down to contact point
        3. Wait
        4. Move back up
    """



    points = load_points_csv(csv_path)

    if not points:
        raise RuntimeError("CSV file contains no points.")

    points = points[:max_points]

    print(f"Connecting to Flexiv robot: {ROBOT_SN}")
    robot = flexivrdk.Robot(ROBOT_SN)

    if robot.fault():
        print("Robot fault detected. Attempting to clear fault...")
        robot.ClearFault()
        time.sleep(2.0)

        if robot.fault():
            raise RuntimeError("Robot fault could not be cleared.")

    print("Enabling robot...")
    robot.Enable()

    while not robot.operational():
        print("Waiting for robot to become operational...")
        time.sleep(1.0)

    print("Robot is operational.")

    print("Switching to primitive execution mode...")
    robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)

    for i, point in enumerate(points):
        safe_pose = canvas_point_to_robot_pose(point, contact=False)
        contact_pose = canvas_point_to_robot_pose(point, contact=True)

        print(f"\nExecuting point {i + 1}/{len(points)}")
        print(f"Safe pose:     {safe_pose}")
        print(f"Contact pose:  {contact_pose}")

        safe_cmd = pose_to_movel_primitive(safe_pose)
        contact_cmd = pose_to_movel_primitive(contact_pose)

        print(f"Command: {safe_cmd}")
        robot.ExecutePrimitive(safe_cmd)
        wait_until_robot_done(robot)

        print(f"Command: {contact_cmd}")
        robot.ExecutePrimitive(contact_cmd)
        wait_until_robot_done(robot)

        # Later:
        # turn tufting gun ON here
        time.sleep(DWELL_TIME_S)
        # turn tufting gun OFF here

        print(f"Command: {safe_cmd}")
        robot.ExecutePrimitive(safe_cmd)
        wait_until_robot_done(robot)

    print("\nFinished robot CSV motion.")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file with X,Y,Z columns",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print converted poses only",
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Move the real robot",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=5,
        help="Maximum number of points to execute",
    )

    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.csv_path, args.max_points)
    elif args.move:
        move_robot_from_csv(args.csv_path, args.max_points)
    else:
        print("Choose one mode:")
        print("  --dry-run")
        print("  --move")


if __name__ == "__main__":
    main()