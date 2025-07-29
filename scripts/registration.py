#!/usr/bin/env python3
from digiforest_registration.tasks.registration import Registration, RegistrationResult
from digiforest_registration.optimization.io import (
    write_tiles_to_pose_graph_file,
    write_aerial_transforms_to_pose_graph_file,
)
from digiforest_registration.utils import CloudIO, is_cloud_name, TileConfigReader
from digiforest_registration.utils import crop_cloud, crop_cloud_to_size
from pathlib import Path
from typing import Tuple
import numpy as np
import os
import open3d as o3d

import argparse
import yaml
import logging


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="cloud_registration",
        description="Registers a MLS cloud to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--config", default=None, help="yaml config file")
    parser.add_argument("--uav_cloud", help="the path to the UAV point cloud")
    parser.add_argument("--mls_cloud", default=None, help="the path to the MLS cloud")
    parser.add_argument(
        "--mls_cloud_folder",
        default=None,
        help="the path to the folder containing the MLS clouds. Set either this parameter or mls_cloud",
    )
    parser.add_argument(
        "--tiles_conf_file", default=None, help="path to the tiles configuration file"
    )
    parser.add_argument(
        "--ground_segmentation_method",
        nargs="?",
        default="default",
        help="Method to use for ground segmentation. Options: default, csf",
    )
    parser.add_argument("--correspondence_matching_method", nargs="?", default="graph")
    parser.add_argument(
        "--mls_feature_extraction_method",
        nargs="?",
        default="canopy_map",
        help="Options : canopy_map or tree_segmentation). It's the method to extract the features of the mls cloud.\
        canopy_map works well if the canopy is visible in the mls cloud. If the canopy is not visible, the other method must be used",
    )
    parser.add_argument(
        "--offset",
        nargs="+",
        type=float,
        default=None,
        help="translation offset to apply to the MLS clouds",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        help="path to the output folder where the transformed MLS clouds and the new pose graph will be stored",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="set this to true to visualize the intermediate steps of the pipeline",
    )
    parser.add_argument(
        "--save_pose_graph", default=False, action="store_true", help="save pose graph"
    )
    parser.add_argument(
        "--pose_graph_file", default=None, help="path to the MLS pose graph file"
    )
    parser.add_argument(
        "--downsample-cloud",
        default=False,
        action="store_true",
        help="downsample input point clouds",
    )
    parser.add_argument("--grid_size_row", type=int, default=0)
    parser.add_argument("--grid_size_col", type=int, default=0)
    parser.add_argument("--min_distance_between_peaks", type=float, default=2.5)
    parser.add_argument("--max_number_of_clique", type=int, default=5)
    parser.add_argument(
        "--crop_mls_cloud",
        default=False,
        action="store_true",
        help="crop the input MLS cloud",
    )
    parser.add_argument(
        "--icp_fitness_score_threshold",
        type=float,
        default=0.85,
        help="threshold of the final ICP step",
    )
    parser.add_argument("--logging_dir", type=str, help="path of the logging directory")
    parser.add_argument(
        "--log_level",
        default="DEBUG",
        help="set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
            for key, value in config.items():
                setattr(args, key, value)

    return args


def check_inputs_validity(args) -> Tuple[str, str, str]:

    mls_cloud_filenames = []
    mls_cloud_folder = None
    uav_cloud_filename = Path(args.uav_cloud)
    if not uav_cloud_filename.exists():
        raise ValueError(f"Input file [{uav_cloud_filename}] does not exist")

    if args.mls_cloud_folder is None and args.mls_cloud is None:
        raise ValueError("Either --mls_cloud or --mls_cloud_folder must be specified")

    if args.mls_cloud is not None:
        mls_cloud_filename = Path(args.mls_cloud)
        mls_cloud_filenames.append(mls_cloud_filename)
        if not mls_cloud_filename.exists():
            raise ValueError(f"Input file [{mls_cloud_filename}] does not exist")

    if args.mls_cloud_folder is not None:
        mls_cloud_folder = Path(args.mls_cloud_folder)
        if not mls_cloud_folder.is_dir():
            raise ValueError(f"Input folder [{mls_cloud_folder}] does not exist")
        else:
            # Get all the ply files in the folder
            for entry in mls_cloud_folder.iterdir():
                if entry.is_file():
                    if is_cloud_name(entry):
                        mls_cloud_filenames.append(entry)

    return mls_cloud_filenames, mls_cloud_folder, uav_cloud_filename


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # set seed for deterministic results
    o3d.utility.random.seed(12345)

    args = parse_inputs()

    # Check validity of inputs
    (
        mls_cloud_filenames,
        mls_cloud_folder,
        uav_cloud_filename,
    ) = check_inputs_validity(args)

    # Logger
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("digiforest_registration")

    # Loading the data
    offset = None
    if args.offset is not None and len(args.offset) == 3:
        offset = np.array(
            [args.offset[0], args.offset[1], args.offset[2]], dtype=np.float32
        )

    cloud_io = CloudIO(offset, logger, args.downsample_cloud)
    uav_cloud = cloud_io.load_cloud(str(uav_cloud_filename))

    if args.output_folder is not None and args.tiles_conf_file is not None:
        tile_config_reader = TileConfigReader(
            args.tiles_conf_file, offset, args.grid_size_col, args.grid_size_row
        )

    # Registration
    failures = []
    successes = []
    registration_results = {}
    for mls_cloud_filename in mls_cloud_filenames:

        logger.info(f"Processing file: {mls_cloud_filename.name}")
        original_mls_cloud = cloud_io.load_cloud(str(mls_cloud_filename))

        # cropping input clouds
        if args.crop_mls_cloud:
            mls_cloud = crop_cloud_to_size(original_mls_cloud, size=30)
        else:
            mls_cloud = original_mls_cloud
        cropped_uav_cloud = crop_cloud(uav_cloud, mls_cloud, padding=20)

        if args.debug:
            mls_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            cropped_uav_cloud.paint_uniform_color([0.0, 1.0, 0])
            o3d.visualization.draw_geometries(
                [mls_cloud.to_legacy()],
                window_name="Initial MLS cloud",
            )
            o3d.visualization.draw_geometries(
                [cropped_uav_cloud.to_legacy()],
                window_name="Initial uav",
            )

        logging_dir = args.logging_dir
        if args.logging_dir is None:
            logging_dir = "./logs"
        logging_dir = os.path.join(logging_dir, mls_cloud_filename.stem)

        registration = Registration(
            cropped_uav_cloud,
            mls_cloud,
            args.ground_segmentation_method,
            args.correspondence_matching_method,
            args.mls_feature_extraction_method,
            args.icp_fitness_score_threshold,
            args.min_distance_between_peaks,
            args.max_number_of_clique,
            logging_dir,
            debug=args.debug,
        )
        success = registration.registration()

        logger.info(f"File: {mls_cloud_filename.name}, {success}")
        if not success:
            failures.append((mls_cloud_filename.name, registration.report))
        else:
            successes.append((mls_cloud_filename.name, registration.report))

        result = RegistrationResult()
        result.transform = registration.transform
        result.success = success
        result.icp_fitness = registration.report["icp_fitness"]
        registration_results[mls_cloud_filename.name] = result

        if args.output_folder is not None:
            output_filename = os.path.join(args.output_folder, mls_cloud_filename.name)
            cloud_io.save_cloud(
                registration.transform_cloud(original_mls_cloud),
                output_filename,
                local_coordinates=False,
            )

    logger.info(f"Total number of failures: {len(failures)}")
    logger.info(f"Total number of clouds: {len(mls_cloud_filenames)}")
    logger.info(f"Failures: {failures}")
    logger.info(f"Successes: {successes}")

    # save registration results
    if args.output_folder is not None:
        import pickle

        output_file_path = os.path.join(args.output_folder, "registration_results.pkl")
        pickle.dump(registration_results, open(output_file_path, "wb"))

    # save pose graph
    if args.tiles_conf_file is not None:

        # saving the pose graph in case we are processing tiles
        if args.save_pose_graph and args.output_folder is not None:
            pose_graph_path = os.path.join(args.output_folder, "pose_graph.g2o")
            write_tiles_to_pose_graph_file(
                mls_cloud_folder,
                pose_graph_path,
                args.grid_size_row,
                args.grid_size_col,
                registration_results,
                args.icp_fitness_score_threshold,
                tile_config_reader,
            )

    elif args.pose_graph_file is not None:
        # processing mls clouds
        if args.save_pose_graph and args.output_folder is not None:
            output_pose_graph_path = os.path.join(args.output_folder, "pose_graph.g2o")
            write_aerial_transforms_to_pose_graph_file(
                Path(args.pose_graph_file),
                output_pose_graph_path,
                registration_results,
                args.icp_fitness_score_threshold,
            )
