#!/usr/bin/env python3
from slam_datasets.carmen.carmen_reader import CarmenLogReader
import math
import statistics

LOG_PATH = "/home/matheus/data/datasets/csail-newcarmen.log/mit-csail-3rd-floor-2005-12-17-run4.log"

reader = CarmenLogReader(LOG_PATH)
stamps = []
scan_sizes = []

for scan in reader.iter_scans():
    stamps.append(scan.stamp)
    scan_sizes.append(len(scan.ranges))

print("Total scans:", len(stamps))

# Timestamp checks
monotonic = all(stamps[i] < stamps[i+1] for i in range(len(stamps)-1))
print("Timestamps strictly increasing:", monotonic)

# Scan size consistency
unique_sizes = set(scan_sizes)
print("Unique scan sizes:", unique_sizes)

# Scan period statistics
if len(stamps) > 1:
    dts = [stamps[i+1] - stamps[i] for i in range(len(stamps)-1)]
    print("Mean scan period (s):", statistics.mean(dts))
    print("Std scan period (s):", statistics.pstdev(dts))
    print("Min / Max scan period:", min(dts), max(dts))

# Basic sanity
print("Any empty scans:", any(n == 0 for n in scan_sizes))
print("Any NaNs:", any(math.isnan(t) for t in stamps))
