#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slam_datasets.carmen.carmen_reader import CarmenLogReader
from slam_datasets.records import Pose2D


@dataclass(frozen=True)
class RelationEdge:
    """Relation edge representing a 2D pose constraint between two timestamps
    in the graph.

    Attributes:
        src_stamp: source timestamp of the constraint
        dst_stamp: destination timestamp of the constraint
        dx: relative x translation from source to destination in meters
        dy: relative y translation from source to destination in meters
        dtheta: relative rotation from source to destination in radians
    """
    src_stamp: float
    dst_stamp: float
    dx: float
    dy: float
    dtheta: float



@dataclass(frozen=True)
class ScanSample:
    """Scan sample representing a 2D laser scan at a specific timestamp and 2D pose.

    Attributes:
        stamp: timestamp of the scan
        pose: robot pose (usually odometry) at the scan timestamp
        angle_min: start angle of the scan
        angle_increment: angle increment between beams
        range_min: minimum valid range value
        ranges: array of range measurements
    """
    stamp: float
    pose: Pose2D
    angle_min: float
    angle_increment: float
    range_min: float
    ranges: np.ndarray


def normalize_angle(angle: float) -> float:
    """Normalize an angle in radians to the range [-pi, pi].
    
    Args:
        angle: input angle in radians
    
    Returns:
        normalized angle in radians
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compose_pose(base: Pose2D, delta: Pose2D) -> Pose2D:
    """Compose two 2D poses: c_M_a = c_M_b ⊕ b_M_a.

    Args:
        base: base pose (c_M_b)
        delta: relative pose (b_M_a)

    Returns:
        composed pose (c_M_a)
    """
    c = math.cos(base.yaw)
    s = math.sin(base.yaw)
    x = base.x + c * delta.x - s * delta.y
    y = base.y + s * delta.x + c * delta.y
    yaw = normalize_angle(base.yaw + delta.yaw)
    return Pose2D(x=x, y=y, yaw=yaw)


def invert_pose(pose: Pose2D) -> Pose2D:
    """Invert a 2D pose: a_M_b = (b_M_a)^-1.

    Args:
        pose: input pose to invert (b_M_a)

    Returns:
        inverted pose (a_M_b)
    """
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    x = -c * pose.x - s * pose.y
    y = s * pose.x - c * pose.y
    yaw = normalize_angle(-pose.yaw)
    return Pose2D(x=x, y=y, yaw=yaw)


def between_pose(a: Pose2D, b: Pose2D) -> Pose2D:
    """Compute the relative pose between two poses: a_M_b = (b_M_a)^-1 ⊕ (b_M_c).

    Args:
        a: first pose (b_M_a)
        b: second pose (b_M_c)
    
    Returns:
        relative pose from a to b (a_M_b)
    """
    return compose_pose(invert_pose(a), b)



def parse_relations(relations_path: Path) -> List[RelationEdge]:
    """Extract edges (pose-graph slam constraints) from file.

    Args:
        relations_path: path to the relations log file, with lines of the form:
            <src_stamp> <dst_stamp> <dx> <dy> <dtheta> ... (other fields are ignored)

    Returns:
        List of RelationEdge objects representing the parsed edges.
    """
    edges: List[RelationEdge] = []
    with relations_path.open("rt", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 8:
                continue
            try:
                edges.append(
                    RelationEdge(
                        src_stamp=float(tokens[0]),
                        dst_stamp=float(tokens[1]),
                        dx=float(tokens[2]),
                        dy=float(tokens[3]),
                        dtheta=float(tokens[7]),
                    )
                )
            except ValueError:
                continue
    return edges



def build_ground_truth_pose_map(
    edges: Sequence[RelationEdge],
) -> Tuple[Dict[float, Pose2D], Dict[float, int]]:
    """Compute the robot path using a Dijkstra-like search over the relation edges.

    The algorithm is as follows:
        1. Order the relation edges timestamps
        2. Create adjacency list relating connected nodes (with pose constraints
        and time index cost)
        3. Run a Dijkstra-like search to find the robot path (sequencial timestamps).

    Args:
        edges: List of relation edges representing pose constraints.

    Returns:
        Tuple of two dictionaries:
        - The 2D pose of the related node in local frame
        - The related node ID
    """
    if not edges:
        return {}, {}

    # Sorte edge stamps and initialize adjacency list
    # TODO: reuse 'relation_stamps' calculated in main() instead of 'stamps'
    stamps = sorted({edge.src_stamp for edge in edges} | {edge.dst_stamp for edge in edges})
    stamp_index = {stamp: idx for idx, stamp in enumerate(stamps)}
    # Adjacency list: stamp -> List of (neighbor_stamp, relative_pose, step_cost)
    adjacency: Dict[float, List[Tuple[float, Pose2D, int]]] = {stamp: [] for stamp in stamps}

    # Browse edges and populate adjacency list with relative pose constraints
    for edge in edges:
        delta = Pose2D(x=edge.dx, y=edge.dy, yaw=edge.dtheta)
        reverse = invert_pose(delta)
        step_cost = abs(stamp_index[edge.dst_stamp] - stamp_index[edge.src_stamp])
        step_cost = max(step_cost, 1) # diff between stamps indeces (min is 1)
        # Add bidirectional adjacencies
        adjacency[edge.src_stamp].append((edge.dst_stamp, delta, step_cost))
        adjacency[edge.dst_stamp].append((edge.src_stamp, reverse, step_cost))

    # Run Dijkstra-like search to find best-cost (i.e. the most sequentially connected path)
    best_cost: Dict[float, int] = {}   # best known path cost (as a time index) for each timestamp
    pose_map: Dict[float, Pose2D] = {} # the computed relative pose for each timestamp
    node_id_map: Dict[float, int] = {} # relates timestamps to a node ID
    node_id = 0                        # current node 
    # First node is the first timestamp ('stamps" is ordered)
    for start in stamps:  
        # jump if node already visited
        if start in best_cost:
            continue

        # Start new search from new unvisited node (cost and dist to itself is 0)
        node_id += 1
        best_cost[start] = 0
        pose_map[start] = Pose2D(x=0.0, y=0.0, yaw=0.0)
        node_id_map[start] = node_id
        queue: List[Tuple[int, float]] = [(0, start)]
        # Browse queue of nodes to visit
        while queue:
            # Take the lowest cost node in the queue.
            # Ignore it if a best cost is known for the node
            cost, stamp = heapq.heappop(queue) # take lowest cost
            if cost > best_cost.get(stamp, math.inf):
                continue

            # Expande neighbors for current node (stamp)
            node_id_map[stamp] = node_id
            base_pose = pose_map[stamp]
            for nxt_stamp, rel_delta, edge_cost in adjacency[stamp]:
                # Ignore neighbor if a better cost (better path) is already known
                new_cost = cost + edge_cost
                if new_cost >= best_cost.get(nxt_stamp, math.inf):
                    continue
                # Update best cost and pose for the neighbor. Add it to the queue
                best_cost[nxt_stamp] = new_cost
                pose_map[nxt_stamp] = compose_pose(base_pose, rel_delta)
                node_id_map[nxt_stamp] = node_id
                heapq.heappush(queue, (new_cost, nxt_stamp))
    return pose_map, node_id_map


def load_scans(raw_log_path: Path) -> List[ScanSample]:
    """Load laser scan data from a raw log file.

    Args:
        raw_log_path: path to the raw CARMEN log file.

    Returns:
        List of ScanSample objects representing the parsed scans with poses.
    """
    scans: List[ScanSample] = []
    reader = CarmenLogReader(raw_log_path)
    for scan in reader.iter_scans():
        if scan.robot_pose is None:
            continue
        scans.append(
            ScanSample(
                stamp=scan.stamp,
                pose=scan.robot_pose,
                angle_min=scan.angle_min,
                angle_increment=scan.angle_increment,
                range_min=scan.range_min,
                ranges=np.asarray(scan.ranges, dtype=np.float32),
            )
        )
    return scans


def determine_scan_subset(
    scans: Sequence[ScanSample],
    relation_stamps: Sequence[float],
    use_all_scans: bool,
    scan_step: int,
    max_scans: int,
) -> List[ScanSample]:
    """Select a subset of scans.
    
    Scans are selected based on:
        1. Whether they have correspondances in the relation timestamps
        2. The downsampling step (scan_step)
        3. The maximum number of scans to use (max_scans)

    Args:
        scans: the input scans to select from
        relation_stamps: sorted list of timestamps that appear in the relations (if any)
        use_all_scans: if True, ignore relation_stamps and use all scans
        scan_step: use every Nth scan after initial filtering (1 = use all)
        max_scans: optional hard limit on the number of scans to use (0 = no limit)
    """

    # Select only scans with correspondances in relation_stamps.
    # Select all scans otherwise.
    selected: List[ScanSample]
    if relation_stamps and not use_all_scans:
        scan_by_stamp = {scan.stamp: scan for scan in scans}
        selected = [scan_by_stamp[stamp] for stamp in sorted(relation_stamps) if stamp in scan_by_stamp]
    else:
        selected = list(scans)

    # Filtering:
    # - keep every scan_step element.
    # - keep the first max_scans elements
    selected = selected[::max(scan_step, 1)]
    if max_scans > 0:
        selected = selected[:max_scans]
    return selected


def compute_grid_bounds(
    scans: Sequence[ScanSample],
    max_range: float,
    padding: float,
) -> Tuple[float, float, float, float]:
    """"Compute the bounds of the occupancy grid based on.

    Computer the occupancy grid map bound based on the scan and
    endpoint poses.

    Args:
        scans: the input scans to use for mapping
        max_range: maximum lidar range used for mapping in meters
        padding: extra map padding in meters around trajectory and endpoints

    Returns:
        (min_x, max_x, min_y, max_y) bounds of the map in world coordinates
    """

    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    # Browse scans and update bounds
    for scan in scans:
        x0 = scan.pose.x
        y0 = scan.pose.y
        min_x = min(min_x, x0)
        max_x = max(max_x, x0)
        min_y = min(min_y, y0)
        max_y = max(max_y, y0)
        
        # Skip scans if any invalid measurement
        valid = np.isfinite(scan.ranges) & (scan.ranges > scan.range_min) & (scan.ranges <= max_range)
        if not np.any(valid):
            continue

        # Get idx of valid beams and compute their endpoints in world frame
        # FIXME: normalize beam angles
        beam_idx = np.flatnonzero(valid)
        beam_angles = scan.pose.yaw + scan.angle_min + scan.angle_increment * beam_idx
        beam_ranges = scan.ranges[beam_idx]
        beam_x = x0 + np.cos(beam_angles) * beam_ranges
        beam_y = y0 + np.sin(beam_angles) * beam_ranges
        
        # Update bounds
        min_x = min(min_x, float(np.min(beam_x)))
        max_x = max(max_x, float(np.max(beam_x)))
        min_y = min(min_y, float(np.min(beam_y)))
        max_y = max(max_y, float(np.max(beam_y)))

    # FIXME: add a validity check method
    if not math.isfinite(min_x):
        min_x, max_x, min_y, max_y = -5.0, 5.0, -5.0, 5.0

    return min_x - padding, max_x + padding, min_y - padding, max_y + padding


def world_to_grid(
    x: float,
    y: float,
    min_x: float,
    min_y: float,
    resolution: float,
) -> Tuple[int, int]:
    """Convert world coordinates to grid coordinates."""
    col = int(round((x - min_x) / resolution))
    row = int(round((y - min_y) / resolution))
    return col, row


def add_ray(
    log_odds: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    mark_occupied_endpoint: bool,
    free_delta: float,
    occ_delta: float,
) -> None:
    """Update the log-odds grid along a ray from (x0, y0) to (x1, y1).

    Uses Bresenham's line algorithm to determine the free and occupied cells
    along the ray connecting (x0, y0) to (x1, y1).

    Args:
        log_odds: 2D array of log-odds values to update in-place
        x0, y0: grid coordinates of the ray origin (scan pose)
        x1, y1: grid coordinates of the ray endpoint
        mark_occupied_endpoint: True if the endpoint cell is marked as occupied; False otherwise
        free_delta: log-odds increment for free cells along the ray
        occ_delta: log-odds increment for the occupied endpoint cell

    Returns:
        None (the log_odds array is updated in-place)
    """
    width = log_odds.shape[1]
    height = log_odds.shape[0]

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x = x0
    y = y0
    while True:
        at_endpoint = x == x1 and y == y1
        if 0 <= x < width and 0 <= y < height:
            if at_endpoint and mark_occupied_endpoint:
                log_odds[y, x] += occ_delta
            else:
                log_odds[y, x] += free_delta

        if at_endpoint:
            break

        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            x += sx
        if err2 < dx:
            err += dx
            y += sy


def build_occupancy_grid(
    scans: Sequence[ScanSample],
    resolution: float,
    max_range: float,
    beam_step: int,
    prob_occ: float,
    prob_free: float,
    log_odds_clip: float,
    padding: float,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Build an occupancy grid from the input scans and poses.
    
    Build an occupancy grid map using Bresenham's raytracing algorithm to
    update log-odds values along scan beams.

    Args:
        scans: the input scans to use for mapping
        resolution: grid resolution in meters
        max_range: maximum lidar range used for mapping in meters
        beam_step: use every Nth beam for mapping (1 = use all beams)
        prob_occ: occupied-cell inverse-model probability
        prob_free: free-cell inverse-model probability
        log_odds_clip: saturation value for log-odds updates
        padding: extra map padding in meters around trajectory and endpoints
    
    Returns:
        occupancy_probability: 2D array of occupancy probabilities in [0, 1]
        extent: (min_x, max_x, min_y, max_y) bounds of the map in world coordinates
    """

    # Compute grid bounds and size based on scan endpoints and robot poses
    min_x, max_x, min_y, max_y = compute_grid_bounds(scans, max_range, padding)
    width = int(math.ceil((max_x - min_x) / resolution)) + 1
    height = int(math.ceil((max_y - min_y) / resolution)) + 1

    # Convert probabilities into log-odds
    log_odds = np.zeros((height, width), dtype=np.float32)
    occ_delta = float(math.log(prob_occ / (1.0 - prob_occ)))
    free_delta = float(math.log(prob_free / (1.0 - prob_free)))

    # Process every scan beam and update grid log-odds using Bresenham's raytracing algorithm.
    step = max(beam_step, 1)
    for scan_idx, scan in enumerate(scans):
        # Get scan origin in grid coordinates
        x0, y0 = world_to_grid(scan.pose.x, scan.pose.y, min_x, min_y, resolution)
        # Browse scan beams
        for beam_idx in range(0, scan.ranges.size, step):
            measured_range = float(scan.ranges[beam_idx])
            # Skip if invalid
            if not math.isfinite(measured_range):
                continue
            if measured_range <= scan.range_min:
                continue
            # Endpoint is occupied (is hit) or not
            is_hit = measured_range < max_range
            used_range = min(measured_range, max_range)
            # FIXME: normalize beam angle
            beam_angle = scan.pose.yaw + scan.angle_min + beam_idx * scan.angle_increment
            x1_world = scan.pose.x + math.cos(beam_angle) * used_range
            y1_world = scan.pose.y + math.sin(beam_angle) * used_range
            x1, y1 = world_to_grid(x1_world, y1_world, min_x, min_y, resolution)
            # Update grid
            add_ray(log_odds, x0, y0, x1, y1, is_hit, free_delta, occ_delta)

        # Clip log-odds every 200 scans to avoid numerical issues. 
        # FIXME: 200 should be a parameter
        if (scan_idx + 1) % 200 == 0:
            np.clip(log_odds, -log_odds_clip, log_odds_clip, out=log_odds)

    # Clip log-odds
    np.clip(log_odds, -log_odds_clip, log_odds_clip, out=log_odds)

    # Convert log-odds to occupancy probabilities
    occupancy_probability = 1.0 / (1.0 + np.exp(-log_odds))
    return occupancy_probability, (min_x, max_x, min_y, max_y)


def align_ground_truth_trajectory(
    pose_map: Dict[float, Pose2D],
    component_map: Dict[float, int],
    odom_by_stamp: Dict[float, Pose2D],
) -> List[List[Pose2D]]:
    """Compute ground-truth trajectory segments aligned to the odometry frame.

    The groud-truth trajectory is computed by chaining the relative poses
    from the odometry origin.

    Args:
    pose_map: A dictionary mapping timestamps to their corresponding 2D poses in the relations frame.
    component_map: A dictionary mapping timestamps to their corresponding component IDs.
    odom_by_stamp: A dictionary mapping timestamps to their corresponding 2D poses in the odometry frame.

    Returns:
    A list of trajectory segments, where each segment is a list of 2D poses representing the ground-truth
    trajectory aligned to the odometry frame.
    """
    # Get trajectory stamps by component ID (if stamp is present in both
    # pose_map and odom_by_stamp).
    stamps_by_component: Dict[int, List[float]] = {}
    for stamp, component_id in component_map.items():
        if stamp in pose_map and stamp in odom_by_stamp:
            stamps_by_component.setdefault(component_id, []).append(stamp)

    segments: List[List[Pose2D]] = []
    for component_id in sorted(stamps_by_component):
        component_stamps = sorted(stamps_by_component[component_id])
        if not component_stamps:
            continue

        # Odom and GT with same timestamp origin
        first_stamp = component_stamps[0]
        gt_origin = pose_map[first_stamp]
        odom_origin = odom_by_stamp[first_stamp]

        # Compute trajectory segments relative to odom origin.
        segment: List[Pose2D] = []
        for stamp in component_stamps:
            rel_from_gt_origin = between_pose(gt_origin, pose_map[stamp])
            segment.append(compose_pose(odom_origin, rel_from_gt_origin))
        segments.append(segment)

    return segments


def split_ground_truth_by_component(
    pose_map: Dict[float, Pose2D],
    component_map: Dict[float, int],
) -> List[List[Pose2D]]:
    """
    Split the ground-truth trajectory into segments based on the connected components.

    This function groups the poses in `pose_map` into segments based on the component IDs.

    Args:
        pose_map: A dictionary mapping timestamps to their corresponding 2D poses.
        component_map: A dictionary mapping timestamps to their corresponding component IDs.
    """
    # Get stamps present in both pose and component maps
    stamps_by_component: Dict[int, List[float]] = {}
    for stamp, component_id in component_map.items():
        if stamp in pose_map:
            stamps_by_component.setdefault(component_id, []).append(stamp)
    # Retrieve segments
    segments: List[List[Pose2D]] = []
    for component_id in sorted(stamps_by_component):
        component_stamps = sorted(stamps_by_component[component_id])
        segments.append([pose_map[stamp] for stamp in component_stamps])
    return segments


def save_plot(
    occupancy_probability: np.ndarray,
    extent: Tuple[float, float, float, float],
    odom_traj: Sequence[Pose2D],
    gt_segments: Sequence[Sequence[Pose2D]],
    output_path: Path,
) -> None:
    min_x, max_x, min_y, max_y = extent
    occupancy_img = 1.0 - occupancy_probability

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(
        occupancy_img,
        cmap="gray",
        origin="lower",
        extent=[min_x, max_x, min_y, max_y],
        vmin=0.0,
        vmax=1.0,
    )

    if odom_traj:
        ax.plot(
            [pose.x for pose in odom_traj],
            [pose.y for pose in odom_traj],
            color="tab:red",
            linewidth=1.0,
            label="Odometry trajectory",
        )

    for idx, segment in enumerate(gt_segments):
        if not segment:
            continue
        ax.plot(
            [pose.x for pose in segment],
            [pose.y for pose in segment],
            color="tab:cyan",
            linewidth=1.5,
            linestyle="--",
            label="Relations (ground-truth) trajectory" if idx == 0 else None,
        )

    if odom_traj:
        ax.scatter(odom_traj[0].x, odom_traj[0].y, c="lime", s=45, label="Start")
        ax.scatter(odom_traj[-1].x, odom_traj[-1].y, c="blue", s=45, label="End")

    ax.set_title("Occupancy Grid Map with Trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an occupancy grid map from a CARMEN raw log and "
            "overlay odometry and relations-based trajectory."
        )
    )
    parser.add_argument(
        "--raw-log",
        required=True,
        type=Path,
        help="Path to raw CARMEN log (.log/.clf).",
    )
    parser.add_argument(
        "--relations-log",
        type=Path,
        default=None,
        help="Path to relations log with ground-truth constraints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("occupancy_grid_with_trajectory.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Grid resolution in meters.",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=12.0,
        help="Maximum lidar range used for mapping in meters.",
    )
    parser.add_argument(
        "--beam-step",
        type=int,
        default=2,
        help="Use every Nth beam for mapping.",
    )
    parser.add_argument(
        "--scan-step",
        type=int,
        default=1,
        help="Use every Nth scan after initial filtering.",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=0,
        help="Optional hard limit on the number of scans (0 = no limit).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=2.0,
        help="Extra map padding in meters around trajectory and endpoints.",
    )
    parser.add_argument(
        "--prob-occ",
        type=float,
        default=0.72,
        help="Occupied-cell inverse-model probability.",
    )
    parser.add_argument(
        "--prob-free",
        type=float,
        default=0.3,
        help="Free-cell inverse-model probability.",
    )
    parser.add_argument(
        "--log-odds-clip",
        type=float,
        default=5.0,
        help="Saturation value for log-odds updates.",
    )
    parser.add_argument(
        "--all-scans",
        action="store_true",
        help="Use all raw scans. By default, only scans that appear in relations are used when relations are given.",
    )
    parser.add_argument(
        "--no-align-ground-truth",
        action="store_true",
        help="Do not align the relations trajectory to the odometry frame before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    # Check input args
    args = parse_args()
    if not args.raw_log.exists():
        raise FileNotFoundError(f"Raw log not found: {args.raw_log}")
    if args.relations_log is not None and not args.relations_log.exists():
        raise FileNotFoundError(f"Relations log not found: {args.relations_log}")

    # Load input scans
    scans = load_scans(args.raw_log)
    if not scans:
        raise RuntimeError("No FLASER/RLASER scans with robot pose were parsed from the raw log.")

    # Compute robot path from (ground truth) from sequentially connected relation edges
    relation_pose_map: Dict[float, Pose2D] = {} # [stamp: pose]
    relation_node_id_map: Dict[float, int] = {} # [stamp: node_id]
    relation_stamps: List[float] = [] # Sorted list of all timestamps
    if args.relations_log is not None:
        # Extract valid edges
        relation_edges = parse_relations(args.relations_log)
        # Extract all timestamps as a sorted list
        relation_stamps = sorted(
            {edge.src_stamp for edge in relation_edges}
            | {edge.dst_stamp for edge in relation_edges}
        )
        relation_pose_map, relation_node_id_map = build_ground_truth_pose_map(relation_edges)

    # Filter scans based on relation stamps, downsampling step and max nb of scans.
    selected_scans = determine_scan_subset(
        scans=scans,
        relation_stamps=relation_stamps,
        use_all_scans=args.all_scans,
        scan_step=args.scan_step,
        max_scans=args.max_scans,
    )
    if not selected_scans:
        raise RuntimeError("No scans selected for mapping after filters.")

    # Build occupancy grid from selected scans and poses.
    # Get map dimensions
    occupancy_probability, map_extent = build_occupancy_grid(
        scans=selected_scans,
        resolution=args.resolution,
        max_range=args.max_range,
        beam_step=args.beam_step,
        prob_occ=args.prob_occ,
        prob_free=args.prob_free,
        log_odds_clip=args.log_odds_clip,
        padding=args.padding,
    )

    # Generate ground-truth trajectory segments from relation edges and poses.
    odom_traj = [scan.pose for scan in selected_scans]
    gt_segments: List[List[Pose2D]] = [] # GT may contain disconnected segments
    if relation_pose_map:
        if args.no_align_ground_truth:
            gt_segments = split_ground_truth_by_component(
                relation_pose_map,
                relation_node_id_map,
            )
        else:
            odom_by_stamp = {scan.stamp: scan.pose for scan in selected_scans}
            gt_segments = align_ground_truth_trajectory(
                relation_pose_map,
                relation_node_id_map,
                odom_by_stamp,
            )

    print(f"gt_segments: {len(gt_segments)}, odom_traj: {len(odom_traj)}")
    for idx, segment in enumerate(gt_segments):
        print(f"  Segment {idx}: {len(segment)} poses, from {segment[0]} to {segment[-1]}")

    # Generate output map
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_plot(occupancy_probability, map_extent, odom_traj, gt_segments, args.output)

    print(f"Loaded scans: {len(scans)}")
    print(f"Scans used for mapping: {len(selected_scans)}")
    print(f"Relations nodes: {len(relation_stamps)}")
    print(f"Relations poses (all components): {len(relation_pose_map)}")
    print(f"Saved occupancy map plot: {args.output}")


if __name__ == "__main__":
    main()
