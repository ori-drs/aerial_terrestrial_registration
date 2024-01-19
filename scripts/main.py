#!/usr/bin/env python3
from digiforest_registration.tasks.registration import Registration, RegistrationResult
from digiforest_registration.optimization.io import write_tiles_to_pose_graph_file
from digiforest_registration.utils import CloudIO, is_cloud_name, TileConfigReader
from digiforest_registration.utils import crop_cloud, crop_cloud_to_size
from pathlib import Path
from typing import Tuple
import numpy as np
import os
import open3d as o3d

import argparse
import yaml


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="cloud_registration",
        description="Registers a frontier cloud to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--config", default=None, help="yaml config file")
    parser.add_argument("--uav_cloud")
    parser.add_argument("--frontier_cloud", default=None)
    parser.add_argument("--frontier_cloud_folder", default=None)
    parser.add_argument(
        "--tiles_conf_file", default=None, help="tiles configuration file"
    )
    parser.add_argument("--ground_segmentation_method", nargs="?", default="default")
    parser.add_argument("--correspondence_matching_method", nargs="?", default="graph")
    parser.add_argument("--offset", nargs="+", type=float, default=None)
    parser.add_argument("--output_folder", default=None)
    parser.add_argument(
        "--debug", default=False, action="store_true", help="debug mode"
    )
    parser.add_argument(
        "--save_pose_graph", default=False, action="store_true", help="save pose graph"
    )
    parser.add_argument(
        "--downsample-cloud",
        default=False,
        action="store_true",
        help="downsample input point clouds",
    )
    parser.add_argument("--grid_size_row", type=int, default=0)
    parser.add_argument("--grid_size_col", type=int, default=0)
    parser.add_argument(
        "--crop_frontier_cloud",
        default=False,
        action="store_true",
        help="crop frontier cloud",
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
            for key, value in config.items():
                setattr(args, key, value)

    return args


def check_inputs_validity(args) -> Tuple[str, str, str]:

    frontier_cloud_filenames = []
    frontier_cloud_folder = None
    uav_cloud_filename = Path(args.uav_cloud)
    if not uav_cloud_filename.exists():
        raise ValueError(f"Input file [{uav_cloud_filename}] does not exist")

    if args.frontier_cloud_folder is None and args.frontier_cloud is None:
        raise ValueError(
            "Either --frontier_cloud or --frontier_cloud_folder must be specified"
        )

    if args.frontier_cloud is not None:
        frontier_cloud_filename = Path(args.frontier_cloud)
        frontier_cloud_filenames.append(frontier_cloud_filename)
        if not frontier_cloud_filename.exists():
            raise ValueError(f"Input file [{frontier_cloud_filename}] does not exist")

    if args.frontier_cloud_folder is not None:
        frontier_cloud_folder = Path(args.frontier_cloud_folder)
        if not frontier_cloud_folder.is_dir():
            raise ValueError(f"Input folder [{frontier_cloud_folder}] does not exist")
        else:
            # Get all the ply files in the folder
            for entry in frontier_cloud_folder.iterdir():
                if entry.is_file():
                    if is_cloud_name(entry):
                        frontier_cloud_filenames.append(entry)

    if (
        (args.output_folder is not None)
        and args.save_pose_graph
        and (args.tiles_conf_file is None)
    ):
        raise ValueError(f"Tiles configuration file must be specified")

    return frontier_cloud_filenames, frontier_cloud_folder, uav_cloud_filename


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # set seed for deterministic results
    o3d.utility.random.seed(12345)

    args = parse_inputs()

    # Check validity of inputs
    (
        frontier_cloud_filenames,
        frontier_cloud_folder,
        uav_cloud_filename,
    ) = check_inputs_validity(args)

    # Loading the data
    offset = None
    if args.offset is not None and len(args.offset) == 3:
        offset = np.array(
            [args.offset[0], args.offset[1], args.offset[2]], dtype=np.float32
        )

    cloud_io = CloudIO(offset, args.downsample_cloud)
    uav_cloud = cloud_io.load_cloud(str(uav_cloud_filename))

    if args.output_folder is not None and args.tiles_conf_file is not None:
        tile_config_reader = TileConfigReader(
            args.tiles_conf_file, offset, args.grid_size_col, args.grid_size_row
        )

    # Registration
    failures = []
    registration_results = {}
    for frontier_cloud_filename in frontier_cloud_filenames:

        print("Processing file: ", frontier_cloud_filename.name)
        frontier_cloud = cloud_io.load_cloud(str(frontier_cloud_filename))

        if args.crop_frontier_cloud:
            frontier_cloud = crop_cloud_to_size(frontier_cloud, size=30)
        cropped_uav_cloud = crop_cloud(uav_cloud, frontier_cloud, padding=20)

        if args.debug:
            frontier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            cropped_uav_cloud.paint_uniform_color([0.0, 1.0, 0])
            o3d.visualization.draw_geometries(
                [frontier_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
                window_name="Initial data",
            )

        registration = Registration(
            cropped_uav_cloud,
            frontier_cloud,
            args.ground_segmentation_method,
            args.correspondence_matching_method,
            debug=args.debug,
        )
        success = registration.registration()

        print("File: ", frontier_cloud_filename.name, success)
        if not success:
            failures.append((frontier_cloud_filename.name, registration.report))

        result = RegistrationResult()
        result.transform = registration.transform
        result.success = success
        registration_results[frontier_cloud_filename.name] = result

        if args.output_folder is not None:
            output_filename = os.path.join(
                args.output_folder, frontier_cloud_filename.name
            )
            cloud_io.save_cloud(
                registration.frontier_cloud_aligned,
                output_filename,
                local_coordinates=True,
            )

    print("Total number of failures: ", len(failures))
    print("Total number of clouds: ", len(frontier_cloud_filenames))
    print("Failures: ", failures)

    # save pose graph
    if args.save_pose_graph and args.output_folder is not None:
        pose_graph_path = os.path.join(args.output_folder, "pose_graph.g2o")
        write_tiles_to_pose_graph_file(
            frontier_cloud_folder,
            pose_graph_path,
            args.grid_size_row,
            args.grid_size_col,
            registration_results,
            tile_config_reader,
        )
