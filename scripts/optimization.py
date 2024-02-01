#!/usr/bin/env python3

from digiforest_registration.optimization.io import load_pose_graph
from digiforest_registration.optimization.optimize_graph import PoseGraphOptimization
from digiforest_registration.utils import CloudIO
from pathlib import Path
import numpy as np
import open3d as o3d

import argparse
import yaml


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="optimization",
        description="Optimize a pose graph",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--config", default=None, help="yaml config file")
    parser.add_argument("--frontier_cloud_folder", default=None)
    parser.add_argument("--pose_graph_file", default=None)
    parser.add_argument("--offset", nargs="+", type=float, default=None)
    parser.add_argument("--output_folder", default=None)
    parser.add_argument(
        "--debug", default=False, action="store_true", help="debug mode"
    )
    parser.add_argument(
        "--load_clouds", default=False, action="store_true", help="show clouds"
    )
    parser.add_argument(
        "--tiles", default=False, action="store_true", help="processing tiles"
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
            for key, value in config.items():
                setattr(args, key, value)
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
    pose_graph = load_pose_graph(
        args.pose_graph_file,
        frontier_cloud_folder,
        cloud_io,
        args.load_clouds,
        args.tiles,
    )

    optimizer = PoseGraphOptimization(
        pose_graph, args.load_clouds, process_tiles=args.tiles
    )
    optimizer.optimize()

    # save the results
    if args.output_folder is not None and args.load_clouds:
        for id, _ in pose_graph.nodes.items():
            try:
                cloud = pose_graph.get_node_cloud(id).clone()

                # Transform the cloud to take into account the factor graph optimization
                initial_node_pose = pose_graph.get_initial_node_pose(id)
                node_pose = pose_graph.get_node_pose(id)
                cloud.transform(
                    node_pose.matrix() @ np.linalg.inv(initial_node_pose.matrix())
                )
                # TODO it's still not the correct transformation
                # center = get_cloud_center(cloud)
                # center_pose = np.eye(4)
                # center_pose[0:3, 3] = center
                # node_pose = pose_graph.get_node_pose(id)
                # cloud.transform(node_pose.matrix() @ np.linalg.inv(center_pose))

                cloud_name = pose_graph.get_node_cloud_name(id)
                cloud_path = Path(args.output_folder) / cloud_name

                cloud_io.save_cloud(cloud, str(cloud_path))
            except Exception:
                pass
