#!/usr/bin/env python3
from digiforest_registration.registration.registration import (
    Registration,
    RegistrationResult,
)
from digiforest_registration.utils import CloudIO, TileConfigReader
from digiforest_registration.utils import (
    crop_cloud,
    crop_cloud_to_size,
    parse_inputs,
    check_registration_inputs_validity,
)
from digiforest_registration.utils import ExperimentLogger
from digiforest_registration.registration.registration_io import (
    save_registered_clouds,
    save_posegraph,
)
import numpy as np
import os
import open3d as o3d
import logging


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
    ) = check_registration_inputs_validity(args)

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
    logging_dir = args.logging_dir
    if args.logging_dir is None:
        logging_dir = "./logs"
    registration_logger = ExperimentLogger(base_dir=logging_dir)
    for mls_cloud_filename in mls_cloud_filenames:

        logger.info(f"Processing file: {mls_cloud_filename.name}")
        original_mls_cloud = cloud_io.load_cloud(str(mls_cloud_filename))

        # cropping input clouds
        if args.crop_mls_cloud:
            mls_cloud = crop_cloud_to_size(original_mls_cloud, size=30)
        else:
            mls_cloud = original_mls_cloud
        cropped_uav_cloud = crop_cloud(uav_cloud, mls_cloud, padding=20)

        registration_logger.set_leaf_logging_folder(mls_cloud_filename.stem)
        registration = Registration(
            cropped_uav_cloud,
            mls_cloud,
            args.ground_segmentation_method,
            args.correspondence_matching_method,
            args.mls_feature_extraction_method,
            args.icp_fitness_score_threshold,
            args.min_distance_between_peaks,
            args.max_number_of_clique,
            registration_logger,
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

        save_registered_clouds(
            cloud_io,
            registration,
            mls_cloud_filename,
            original_mls_cloud,
            args.mls_registered_cloud_folder,
            offset,
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
    save_posegraph(
        noise_matrix,
        args.tiles_conf_file,
        args.save_pose_graph,
        args.mls_registered_cloud_folder,
        tile_config_reader,
        args.pose_graph_file,
        registration_results,
        mls_cloud_folder,
        offset,
        args.icp_fitness_score_threshold,
    )
