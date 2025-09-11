import os
from pathlib import Path
import numpy as np

from digiforest_registration.optimization.io import (
    write_tiles_to_pose_graph_file,
    write_aerial_transforms_to_pose_graph_file,
)
from digiforest_registration.utils import CloudIO, TileConfigReader


def save_registered_clouds(
    cloud_io: CloudIO,
    registration_module,
    cloud_filename: Path,
    original_cloud,
    output_folder,
):
    """
    Save registered clouds.

    Args:
        cloud_io: CloudIO instance.
        original_cloud: The original MLS cloud.
        transformed_cloud: The transformed MLS cloud.
        output_folder: Folder to save clouds.
        filename_stem: Stem of the filename.
        filename_suffix: Suffix of the filename (e.g., '.ply').
        offset: Optional offset array.
    """

    if output_folder is None:
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = os.path.join(output_folder, cloud_filename.name)
    cloud_io.save_cloud(
        registration_module.transform_cloud(original_cloud),
        output_filename,
        local_coordinates=False,
    )


def save_posegraph(
    noise_matrix: np.ndarray,
    tiles_conf_file: str,
    save_pose_graph: bool,
    mls_registered_cloud_folder: str,
    pose_graph_file: str,
    registration_results: dict,
    mls_cloud_folder: Path,
    offset: np.ndarray,
    icp_fitness_score_threshold: float,
):

    if noise_matrix.size != 21:
        raise ValueError(
            "Noise matrix must be a list of 21 floats representing the upper triangular matrix of the 6x6 covariance matrix"
        )
    if tiles_conf_file is not None:
        tile_config_reader = TileConfigReader(tiles_conf_file, offset)
        # saving the pose graph in case we are processing tiles
        if save_pose_graph and mls_registered_cloud_folder is not None:
            pose_graph_path = os.path.join(
                mls_registered_cloud_folder, "pose_graph.g2o"
            )
            write_tiles_to_pose_graph_file(
                mls_cloud_folder,
                pose_graph_path,
                registration_results,
                tile_config_reader,
                offset,
                noise_matrix,
            )

    elif pose_graph_file is not None:
        # processing mls clouds
        if save_pose_graph and mls_registered_cloud_folder is not None:
            output_pose_graph_path = os.path.join(
                mls_registered_cloud_folder, "pose_graph.g2o"
            )

            write_aerial_transforms_to_pose_graph_file(
                Path(pose_graph_file),
                output_pose_graph_path,
                registration_results,
                icp_fitness_score_threshold,
                noise_matrix,
            )
