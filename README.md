# slam_datasets

**slam_datasets** is a lightweight Python package providing dataset
readers and common record types for SLAM datasets. It is designed for
**offline processing, evaluation, and tooling**, with **no ROS
dependency**.

The goal is to make it easy to parse popular SLAM dataset formats into
simple, typed Python structures that can be reused in research code,
benchmarking scripts, or dataset conversion pipelines.

------------------------------------------------------------------------

## Features

-   CARMEN log file parsing (`.clf`, `.log`, optionally `.gz`)
-   Simple Python dataclasses for poses and laser scans
-   Iterator-based API for memory-efficient processing
-   ROS-agnostic (usable in pure Python environments)
-   Compatible with ROS2 dataset players or converters

------------------------------------------------------------------------

## Supported datasets

### CARMEN logs

Currently implemented:

-   `FLASER` / `RLASER` records\
-   Timestamped 2D laser scans\
-   Basic pose extraction when available

Angle metadata is inferred from the number of beams assuming a 180° FOV
when not explicitly specified in the log.

------------------------------------------------------------------------

## Installation

### As a ROS2 package (ament_python)

``` bash
cd <your_ros2_ws>/src
git clone <repo_url>
cd ..
colcon build --packages-select slam_datasets
source install/setup.bash
```

------------------------------------------------------------------------

### As a pure Python package

``` bash
pip install -e .
```

------------------------------------------------------------------------

## Package structure

    slam_datasets/
    │
    ├── slam_datasets/
    │   ├── records.py
    │   ├── carmen/
    │   │   └── carmen_reader.py
    │   └── scripts/
    │       └── validate_carmen_log.py
    │
    ├── test/
    ├── package.xml
    └── setup.py

------------------------------------------------------------------------

## Core modules

### `records.py`

Defines reusable dataclasses:

-   `Pose2D`
-   `LaserScan2DRecord`

These are simple containers to avoid framework lock-in.

------------------------------------------------------------------------

### `carmen/carmen_reader.py`

Main entry point for CARMEN logs.

**Class: `CarmenLogReader`**

Responsibilities:

-   Open plain or gzipped CARMEN logs\
-   Parse laser records\
-   Yield `LaserScan2DRecord` instances

------------------------------------------------------------------------

## Quick start

``` python
from slam_datasets.carmen.carmen_reader import CarmenLogReader

reader = CarmenLogReader(
    "/path/to/log.clf",
    scan_frame_id="laser"
)

for scan in reader.iter_scans():
    print(
        scan.stamp,
        len(scan.ranges),
        scan.angle_min,
        scan.angle_max
    )
```

------------------------------------------------------------------------

## Example use cases

-   Offline SLAM benchmarking\
-   Converting CARMEN → ROS bag\
-   Dataset validation and inspection\
-   Teaching and prototyping SLAM algorithms\
-   Feeding custom SLAM front-ends/back-ends

------------------------------------------------------------------------

## Validation script

``` bash
python -m slam_datasets.scripts.validate_carmen_log <path_to_log>
```

Helps verify format correctness and scan counts.

------------------------------------------------------------------------

## Development

Build and test:

``` bash
colcon build --packages-select slam_datasets
colcon test --packages-select slam_datasets
```

Lint tests may include:

-   flake8\
-   pep257

------------------------------------------------------------------------
