from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

@dataclass(frozen=True)
class LaserScan2DRecord:
    stamp: float          # seconds
    frame_id: str
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: List[float]
    robot_pose: Optional[Pose2D] = None
    laser_pose: Optional[Pose2D] = None
    tv: Optional[float] = None
    rv: Optional[float] = None

