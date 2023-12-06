#!/usr/bin/env python3

from digiforest_registration.optimization.io import load_pose_graph
from digiforest_registration.optimization.optimize_graph import PoseGraphOptimization
from digiforest_registration.utils import CloudIO
from pathlib import Path
import numpy as np
import open3d as o3d

import argparse


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="optimization",
        description="Optimize a pose graph",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--frontier_cloud_folder", default=None)
    parser.add_argument("--pose_graph_file", default=None)
    parser.add_argument("--offset", nargs="+", type=float, default=None)
    parser.add_argument("--output_folder", default=None)
    parser.add_argument(
        "--debug", default=False, action="store_true", help="debug mode"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # set seed for deterministic results
    o3d.utility.random.seed(12345)

    args = parse_inputs()

    # Check validity of inputs
    frontier_cloud_filenames = []

    if args.frontier_cloud_folder is None:
        raise ValueError("--frontier_cloud_folder must be specified")

    if args.pose_graph_file is None:
        raise ValueError("--pose_graph_file must be specified")

    frontier_cloud_folder = Path(args.frontier_cloud_folder)
    if not frontier_cloud_folder.is_dir():
        raise ValueError(f"Input folder [{frontier_cloud_folder}] does not exist")
    else:
        # Get all the ply files in the folder
        for entry in frontier_cloud_folder.iterdir():
            if entry.is_file():
                if entry.suffix == ".ply":
                    frontier_cloud_filenames.append(entry)

    # data loader
    offset = None
    if args.offset is not None and len(args.offset) == 3:
        offset = np.array(
            [args.offset[0], args.offset[1], args.offset[2]], dtype=np.float32
        )

    cloud_io = CloudIO(offset)

    # load the pose graph
    pose_graph = load_pose_graph(args.pose_graph_file, frontier_cloud_folder, cloud_io)

    optimizer = PoseGraphOptimization(pose_graph)
    optimizer.optimize()
