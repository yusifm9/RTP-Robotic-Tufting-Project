"""
for now, pseudo-code. Going to be adjusted once connected to the arm. 

Units:
- CSV input: millimeters
- Robot motion: meters

update ROBOT_SN and calibrate ROBOT_BASE_ORIGIN_M before real tufting.
"""


ROBOT_SN = "Rizon4-XXXXXX"

# Unit conversion
MM_TO_M = 0.001


ROBOT_BASE_ORIGIN_M = {
    "x": 0.400,
    "y": 0.000,
    "z": 0.300,
}

# Safe height above canvas, in meters.
SAFE_Z_OFFSET_M = 0.050

# Contact height offset, in meters.
# Keep this 0.000 until the robot frame is confirmed.
CONTACT_Z_OFFSET_M = 0.000

# Tool orientation quaternion: [qw, qx, qy, qz]
# Placeholder only.
# Later this must match tufting-gun orientation.
DEFAULT_TCP_QUAT = [1.0, 0.0, 0.0, 0.0]

# Slow speed for first  tests
MAX_VEL = 0.003

# Pause at each contact point.
DWELL_TIME_S = 0.20