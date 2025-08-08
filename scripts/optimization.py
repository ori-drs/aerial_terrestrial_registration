#!/usr/bin/env python3

from digiforest_registration.optimization.io import load_pose_graph, write_pose_graph
from digiforest_registration.optimization.optimize_graph import PoseGraphOptimization
from digiforest_registration.utils import CloudIO, parse_inputs
from pathlib import Path
import numpy as np
import open3d as o3d
import logging


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # set seed for deterministic results
    o3d.utility.random.seed(12345)

    args = parse_inputs()

    # Check validity of inputs
    mls_cloud_filenames = []

    if args.mls_registered_cloud_folder is None:
        raise ValueError("--mls_registered_cloud_folder must be specified")

    if args.optimized_cloud_output_folder is None:
        raise ValueError("--optimized_cloud_output_folder must be specified")

    pose_graph_file = str(Path(args.mls_registered_cloud_folder) / "pose_graph.g2o")

    mls_cloud_folder = Path(args.mls_cloud_folder)
    if not mls_cloud_folder.is_dir():
        raise ValueError(f"Input folder [{mls_cloud_folder}] does not exist")
    else:
        # Get all the ply files in the folder
        for entry in mls_cloud_folder.iterdir():
            if entry.is_file():
                if entry.suffix == ".ply":
                    mls_cloud_filenames.append(entry)

    # Logger
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("digiforest_registration")

    # data loader
    offset = None
    if args.offset is not None and len(args.offset) == 3:
        offset = np.array(
            [args.offset[0], args.offset[1], args.offset[2]], dtype=np.float32
        )
    cloud_io = CloudIO(offset)

    # load the pose graph
    # initial_transform = None
    # if args.initial_transform is not None and len(args.initial_transform) == 16:
    #     m = args.initial_transform
    #     initial_transform = np.array(
    #         [
    #             [m[0], m[1], m[2], m[3]],
    #             [m[4], m[5], m[6], m[7]],
    #             [m[8], m[9], m[10], m[11]],
    #             [m[12], m[13], m[14], m[15]],
    #         ],
    #         dtype=np.float32,
    #     )

    pose_graph = load_pose_graph(
        pose_graph_file,
        mls_cloud_folder,
        cloud_io,
        args.load_clouds,
        args.tiles,
    )

    optimizer = PoseGraphOptimization(pose_graph, args.debug, args.load_clouds, logger)
    optimizer.optimize()

    # save the results
    if args.optimized_cloud_output_folder is not None:
        pose_graph_output_file = (
            Path(args.optimized_cloud_output_folder) / "optimized_pose_graph.g2o"
        )
        write_pose_graph(pose_graph, str(pose_graph_output_file))

    if args.optimized_cloud_output_folder is not None and args.load_clouds:
        for id, _ in pose_graph.nodes.items():
            try:
                cloud = pose_graph.get_node_cloud(id).clone()

                # Transform the cloud to take into account the factor graph optimization
                initial_node_pose = pose_graph.get_initial_node_pose(id)
                node_pose = pose_graph.get_node_pose(id)
                cloud.transform(
                    node_pose.matrix() @ np.linalg.inv(initial_node_pose.matrix())
                )

                cloud_name = pose_graph.get_node_cloud_name(id)
                cloud_path = Path(args.optimized_cloud_output_folder) / cloud_name

                cloud_io.save_cloud(cloud, str(cloud_path))
            except Exception:
                pass
