#!/usr/bin/env python3
from digiforest_registration.registration.registration import (
    Registration,
    RegistrationResult,
)
from digiforest_registration.optimization.io import (
    write_tiles_to_pose_graph_file,
    write_aerial_transforms_to_pose_graph_file,
)
from digiforest_registration.utils import CloudIO, is_cloud_name, TileConfigReader
from digiforest_registration.utils import crop_cloud, crop_cloud_to_size, parse_inputs
from pathlib import Path
from typing import Tuple
import numpy as np
import os
import open3d as o3d
import logging


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
    if args.offset is not None and len(args.offset) == 3:
        offset = np.array(
            [args.offset[0], args.offset[1], args.offset[2]], dtype=np.float32
        )
    else:
        # default offset
        offset = np.array([0, 0, 0], dtype=np.float32)

    cloud_io = CloudIO(offset, logger, args.downsample_cloud)
    uav_cloud = cloud_io.load_cloud(str(uav_cloud_filename))

    if (
        args.mls_registered_cloud_folder is not None
        and args.tiles_conf_file is not None
    ):
        tile_config_reader = TileConfigReader(args.tiles_conf_file, offset)

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
            correspondence_graph_distance_threshold=args.correspondence_graph_distance_threshold,
            maximum_rotation_offset=args.maximum_rotation_offset,
            debug=args.debug,
        )
        success = registration.registration()

        logger.info(f"File: {mls_cloud_filename.name}, {success}")
        if not success:
            failures.append(
                (mls_cloud_filename.name, (registration.best_icp_fitness_score))
            )
        else:
            successes.append(
                (mls_cloud_filename.name, (registration.best_icp_fitness_score))
            )

        result = RegistrationResult()
        result.transform = registration.transform
        result.success = success
        result.icp_fitness = registration.best_icp_fitness_score
        registration_results[mls_cloud_filename.name] = result

        if args.mls_registered_cloud_folder is not None:
            if args.offset is not None:
                # an offset is set
                output_filename_with_offset = os.path.join(
                    args.mls_registered_cloud_folder,
                    mls_cloud_filename.stem
                    + "_with_offset"
                    + mls_cloud_filename.suffix,
                )
                cloud_io.save_cloud(
                    registration.transform_cloud(original_mls_cloud),
                    output_filename_with_offset,
                    local_coordinates=True,
                )
            output_filename = os.path.join(
                args.mls_registered_cloud_folder, mls_cloud_filename.name
            )
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
    if args.mls_registered_cloud_folder is not None:
        import pickle

        output_file_path = os.path.join(
            args.mls_registered_cloud_folder, "registration_results.pkl"
        )
        pickle.dump(registration_results, open(output_file_path, "wb"))

    # save pose graph
    noise_matrix = np.array(args.noise_matrix, dtype=np.float32)
    if noise_matrix.size != 21:
        raise ValueError(
            "Noise matrix must be a list of 21 floats representing the upper triangular matrix of the 6x6 covariance matrix"
        )
    if args.tiles_conf_file is not None:

        # saving the pose graph in case we are processing tiles
        if args.save_pose_graph and args.mls_registered_cloud_folder is not None:
            pose_graph_path = os.path.join(
                args.mls_registered_cloud_folder, "pose_graph.g2o"
            )
            write_tiles_to_pose_graph_file(
                mls_cloud_folder,
                pose_graph_path,
                registration_results,
                tile_config_reader,
                offset,
                noise_matrix,
            )

    elif args.pose_graph_file is not None:
        # processing mls clouds
        if args.save_pose_graph and args.mls_registered_cloud_folder is not None:
            output_pose_graph_path = os.path.join(
                args.mls_registered_cloud_folder, "pose_graph.g2o"
            )

            write_aerial_transforms_to_pose_graph_file(
                Path(args.pose_graph_file),
                output_pose_graph_path,
                registration_results,
                args.icp_fitness_score_threshold,
                noise_matrix,
            )
