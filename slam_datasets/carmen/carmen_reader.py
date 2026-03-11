from __future__ import annotations
import gzip
from pathlib import Path
from typing import Iterator, Optional, TextIO, Union

from slam_datasets.records import CarmenRecord, LaserScan2DRecord, Odometry2DRecord, Pose2D

def _open_text(path: Union[str, Path]) -> TextIO:
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", errors="ignore")
    return p.open("rt", errors="ignore")

class CarmenLogReader:
    def __init__(
        self,
        path: Union[str, Path],
        scan_frame_id: str = "laser",
        prefer_message_timestamp: bool = True,
    ) -> None:
        self._path = Path(path)
        self._scan_frame_id = scan_frame_id
        self._prefer_msg_ts = prefer_message_timestamp

    # Iterator over the CARMEN log lines and yield records sequentially.
    def iter_records(self) -> Iterator[CarmenRecord]:
        with _open_text(self._path) as f:
            for line in f:
                rec = self._parse_line(line)
                if rec is not None:
                    yield rec

    # Iterator over the CARMEN log lines and yield only laser scans sequentially.
    def iter_scans(self) -> Iterator[LaserScan2DRecord]:
        for rec in self.iter_records():
            if isinstance(rec, LaserScan2DRecord):
                yield rec

    def _parse_line(self, line: str) -> Optional[CarmenRecord]:
        if not line or line[0] == "#":
            return None

        if line.startswith("ROBOTLASER"):
            return self._parse_robotlaser(line)
        if line.startswith("FLASER "):
            return self._parse_flaser(line)
        if line.startswith("RLASER "):
            return self._parse_rlaser(line)
        if line.startswith("ODOM "):
            return self._parse_odom(line)
        return None

    def _parse_robotlaser(self, line: str) -> Optional[LaserScan2DRecord]:
        tok = line.strip().split()
        try:
            # Header
            idx = 1  # "ROBOTLASER*"

            _laser_type = int(tok[idx]); idx += 1
            start_angle = float(tok[idx]); idx += 1
            _fov = float(tok[idx]); idx += 1
            ang_res = float(tok[idx]); idx += 1
            max_range = float(tok[idx]); idx += 1
            _accuracy = float(tok[idx]); idx += 1
            _remission_mode = int(tok[idx]); idx += 1

            n = int(tok[idx]); idx += 1 # number of readings
            ranges = list(map(float, tok[idx:idx+n])); idx += n

            # After ranges, CARMEN variants differ:
            # Variant A: num_remissions then remissions (often 0), then poses...
            # Variant B: tooclose flags (n ints) then num_remissions then remissions...
            #
            # We detect Variant B by checking whether the next token is an int 0/1 repeated n times.
            # A cheap heuristic: if remaining tokens are too many for Variant A, assume Variant B.
            remaining = len(tok) - idx

            # Variant A minimum tail size:
            # num_rem(1) + laser_pose(3) + robot_pose(3) + tv(1) + rv(1)
            # + forward/side/turn(3) + ipc_ts(1) + host(1) + logger_ts(1) = 15
            min_tail_A = 15

            if remaining > min_tail_A + n:
                # Likely Variant B: skip "tooclose" flags (n ints)
                idx += n

            # num remissions + remissions
            num_rem = int(tok[idx]); idx += 1
            idx += num_rem  # remissions values if present (0 => none)

            # poses
            laser_pose = Pose2D(float(tok[idx]), float(tok[idx+1]), float(tok[idx+2])); idx += 3
            robot_pose = Pose2D(float(tok[idx]), float(tok[idx+1]), float(tok[idx+2])); idx += 3

            tv = float(tok[idx]); idx += 1
            rv = float(tok[idx]); idx += 1

            # safety + turn axis
            idx += 3  # forward_safety, side_safety, turn_axis

            # Trailing logger fields: ipc_timestamp ipc_hostname logger_timestamp
            # Use ipc_timestamp as the scan stamp.
            stamp = float(tok[idx]); idx += 1
            # host = tok[idx]; idx += 1
            # logger_ts = tok[idx]; idx += 1

            return LaserScan2DRecord(
                stamp=stamp,
                frame_id=self._scan_frame_id,
                angle_min=start_angle,
                angle_increment=ang_res,
                range_min=0.0,
                range_max=max_range,
                ranges=ranges,
                robot_pose=robot_pose,
                laser_pose=laser_pose,
                tv=tv,
                rv=rv,
            )
        except Exception:
            return None

    def _parse_flaser(self, line: str) -> Optional[LaserScan2DRecord]:
        """
        Parse a FLASER record from a CARMEN log line.

        Format (common): FLASER <n> <r0..r(n-1)> <x> <y> <theta> <odom_x> <odom_y> <odom_theta> <timestamp> <host> <logger_ts>
        Some variants have extra fields; we parse conservatively from the end.
        """
        tok = line.strip().split()
        try:
            idx = 0
            idx += 1  # "FLASER"
            n = int(tok[idx]); idx += 1
            ranges = list(map(float, tok[idx:idx+n])); idx += n

            # Next 3 are usually laser pose in world.
            x = float(tok[idx]); y = float(tok[idx+1]); th = float(tok[idx+2])
            idx += 3

            # Many logs include odom pose next (x y th).
            # Keep laser and robot poses separately when available.
            laser_pose = Pose2D(x, y, th)
            robot_pose = None
            if (len(tok) - idx) >= 6:
                robot_pose = Pose2D(float(tok[idx]), float(tok[idx + 1]), float(tok[idx + 2]))
                idx += 3
            elif (len(tok) - idx) >= 3:
                robot_pose = Pose2D(x, y, th)

            # Timestamp is at the end: "... ipc_timestamp ipc_hostname logger_timestamp"
            # The line header says: message_name [contents] ipc_timestamp ipc_hostname logger_timestamp
            stamp = float(tok[-3])

            # FIXME: set this as input params
            # FLASER does not embed angle metadata; we need defaults per dataset.
            # Typical SICK LMS: 180 deg FOV with 0.5 deg or 1 deg resolution.
            # We'll make these configurable later; for now infer from n assuming 180°.
            import math
            angle_min = -math.pi / 2.0
            angle_max = math.pi / 2.0
            angle_increment = (angle_max - angle_min) / max(n - 1, 1)

            return LaserScan2DRecord(
                stamp=stamp,
                frame_id=self._scan_frame_id,
                angle_min=angle_min,
                angle_increment=angle_increment,
                range_min=0.0,
                range_max=max(ranges) if ranges else 0.0,
                ranges=ranges,
                robot_pose=robot_pose,
                laser_pose=laser_pose,
            )
        except Exception:
            return None

    def _parse_rlaser(self, line: str) -> Optional[LaserScan2DRecord]:
        # Usually same structure as FLASER
        return self._parse_flaser(line)

    def _parse_odom(self, line: str) -> Optional[Odometry2DRecord]:
        # Common format: ODOM x y theta tv rv accel <timestamp> <host> <logger_timestamp>
        tok = line.strip().split()
        try:
            if len(tok) < 8:
                return None

            x = float(tok[1])
            y = float(tok[2])
            yaw = float(tok[3])
            tv = float(tok[4]) if len(tok) > 4 else None
            rv = float(tok[5]) if len(tok) > 5 else None
            accel = float(tok[6]) if len(tok) > 6 else None
            stamp = float(tok[-3]) if len(tok) >= 10 else float(tok[-1])

            return Odometry2DRecord(
                stamp=stamp,
                pose=Pose2D(x=x, y=y, yaw=yaw),
                tv=tv,
                rv=rv,
                accel=accel,
            )
        except Exception:
            return None
