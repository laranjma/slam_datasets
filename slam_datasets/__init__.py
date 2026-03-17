"""slam_datasets package."""

# so that one can 'from slam_datasets import GraphNode, RelationEdge, ScanSample'
from .mapping_lib import GraphNode, RelationEdge, ScanSample

# so that one can 'from slam_datasets import *'
__all__ = ["GraphNode", "RelationEdge", "ScanSample"]
