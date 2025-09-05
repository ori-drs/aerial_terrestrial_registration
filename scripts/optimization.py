#!/usr/bin/env python3

from digiforest_registration.optimization.io import (
    load_pose_graph,
    write_pose_graph,
    save_optimized_pointclouds,
)
from digiforest_registration.optimization.graph_optimization import (
    PoseGraphOptimization,
)
from digiforest_registration.utils import (
    CloudIO,
    parse_inputs,
    check_optimization_inputs_validity,
)
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
    mls_cloud_folder = check_optimization_inputs_validity(args)
    pose_graph_file = str(Path(args.mls_registered_cloud_folder) / "pose_graph.g2o")

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

    # save the optmized clouds
    save_optimized_pointclouds(
        args.optimized_cloud_output_folder, args.load_clouds, pose_graph, cloud_io
    )
