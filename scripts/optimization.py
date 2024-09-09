#!/usr/bin/env python3

from digiforest_registration.optimization.io import load_pose_graph, write_pose_graph
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
    parser.add_argument("--initial_transform", nargs="+", type=float, default=None)
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
    initial_transform = None
    if args.initial_transform is not None and len(args.initial_transform) == 16:
        m = args.initial_transform
        initial_transform = np.array(
            [
                [m[0], m[1], m[2], m[3]],
                [m[4], m[5], m[6], m[7]],
                [m[8], m[9], m[10], m[11]],
                [m[12], m[13], m[14], m[15]],
            ],
            dtype=np.float32,
        )

    pose_graph = load_pose_graph(
        args.pose_graph_file,
        frontier_cloud_folder,
        cloud_io,
        args.load_clouds,
        args.tiles,
    )

    optimizer = PoseGraphOptimization(
        pose_graph, args.debug, args.load_clouds, process_tiles=args.tiles
    )
    optimizer.optimize()

    # save the results
    if args.output_folder is not None:
        pose_graph_output_file = Path(args.output_folder) / "optimized_pose_graph.g2o"
        write_pose_graph(pose_graph, str(pose_graph_output_file))

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
                # if args.tiles:
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

    # get statistics about how much the graph moved
    # displacements = []
    # displacements_xy = []
    # for id, _ in pose_graph.nodes.items():
    #     initial_node_pose = pose_graph.get_initial_node_pose(id)
    #     transformed_pose_node = gtsam.Pose3(initial_transform)*initial_node_pose
    #     node_pose = pose_graph.get_node_pose(id)
    #     displacements.append(np.linalg.norm(node_pose.matrix()[0:3, 3] - transformed_pose_node.matrix()[0:3, 3]))

    # print(f"Average node displacement {np.mean(displacements)} meters, max node displacement {np.max(displacements)} meters, index {np.argmax(displacements)}")
    # print(f"Min node displacement {np.min(displacements)} meters")
