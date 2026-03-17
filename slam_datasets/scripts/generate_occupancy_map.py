#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slam_datasets.carmen.carmen_reader import CarmenLogReader
from slam_datasets.records import Pose2D
from slam_datasets.mapping_lib import *

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
    relation_segment_id_map: Dict[float, int] = {} # [stamp: segment_id]
    relation_stamps: List[float] = [] # Sorted list of all timestamps
    if args.relations_log is not None:
        # Extract valid edges
        relation_edges = parse_relations(args.relations_log)
        # Extract all timestamps as a sorted list
        relation_stamps = sorted(
            {edge.src_stamp for edge in relation_edges}
            | {edge.dst_stamp for edge in relation_edges}
        )
        relation_pose_map, relation_segment_id_map = build_ground_truth_pose_map(relation_edges,
                                                                                 relation_stamps)

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
    

    # Extract the largest trajectory
    main_pose_map = extract_largest_trajectory(relation_pose_map, relation_segment_id_map)
    # Build pose graph map for the largest trajectory
    pose_graph_map = build_pose_graph_map(main_pose_map, selected_scans, relation_stamps)

    # Generate ground-truth trajectory segments from relation edges and poses.
    odom_traj = [scan.pose for scan in selected_scans]
    gt_segments: List[List[Pose2D]] = [] # GT may contain disconnected segments
    final_gt_pose_graph_map: List[GraphNode] = [] # GT pose graph nodes
    if relation_pose_map:
        if args.no_align_ground_truth:
            gt_segments = split_ground_truth_by_component(
                relation_pose_map,
                relation_segment_id_map,
            )
        else:
            odom_by_stamp = {scan.stamp: scan.pose for scan in selected_scans}
            gt_segments = align_ground_truth_trajectory(
                relation_pose_map,
                relation_segment_id_map,
                odom_by_stamp,
            )
            final_gt_pose_graph_map = align_pose_graph_to_odom_trajectory(pose_graph_map,
                                                                          odom_by_stamp)

    occupancy_probability, map_extent = build_occupancy_grid_map(
        final_gt_pose_graph_map,
        resolution=args.resolution,
        max_range=args.max_range,
        beam_step=args.beam_step,
        prob_occ=args.prob_occ,
        prob_free=args.prob_free,
        log_odds_clip=args.log_odds_clip,
        padding=args.padding,
    )

    # Generate output map
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_plot(occupancy_probability,
              map_extent,
              odom_traj,
              [from_graph_nodes_to_trajectory(final_gt_pose_graph_map)],
              args.output)

    print(f"Loaded scans: {len(scans)}")
    print(f"Scans used for mapping: {len(selected_scans)}")
    print(f"Relations nodes: {len(relation_stamps)}")
    print(f"Relations poses (all components): {len(relation_pose_map)}")
    print(f"Saved occupancy map plot: {args.output}")


if __name__ == "__main__":
    main()
