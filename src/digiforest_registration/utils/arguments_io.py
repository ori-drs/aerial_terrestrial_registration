import yaml
import argparse
from pathlib import Path
from typing import Tuple
from digiforest_registration.utils import is_cloud_name


def check_registration_inputs_validity(args) -> Tuple[str, str, str]:

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


def check_optimization_inputs_validity(args) -> str:

    if args.mls_registered_cloud_folder is None:
        raise ValueError("--mls_registered_cloud_folder must be specified")

    if args.optimized_cloud_output_folder is None:
        raise ValueError("--optimized_cloud_output_folder must be specified")

    mls_cloud_folder = Path(args.mls_cloud_folder)
    if not mls_cloud_folder.is_dir():
        raise ValueError(f"Input folder [{mls_cloud_folder}] does not exist")

    return mls_cloud_folder


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="registration_pipeline",
        description="Registers MLS clouds to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "--config", default=None, help="Path to the yaml configuration file"
    )
    parser.add_argument("--uav_cloud", help="Path to the UAV point cloud")
    parser.add_argument("--mls_cloud", default=None, help="Path to the MLS cloud")
    parser.add_argument(
        "--mls_cloud_folder",
        default=None,
        help="Path to the folder containing the MLS clouds. Set either this parameter or mls_cloud",
    )
    parser.add_argument("--optimized_cloud_output_folder", default=None)
    parser.add_argument(
        "--load_clouds",
        default=False,
        action="store_true",
        help="During the optimization, load and transform the MLS clouds with the pose graph",
    )
    parser.add_argument(
        "--tiles", default=True, action="store_true", help="Processing tiles"
    )
    parser.add_argument(
        "--tiles_conf_file", default=None, help="Path to the tiles configuration file"
    )
    parser.add_argument(
        "--ground_segmentation_method",
        nargs="?",
        default="default",
        help="Method to use for ground segmentation. Options: default, csf",
    )
    parser.add_argument(
        "--correspondence_matching_method",
        nargs="?",
        default="graph",
        help="Method to use for correspondence matching. Options: graph.",
    )
    parser.add_argument(
        "--mls_feature_extraction_method",
        nargs="?",
        default="canopy_map",
        help="Options : canopy_map or tree_segmentation. It's the method used to extract the features of the mls cloud.\
        canopy_map works well if the canopy is visible in the mls cloud. If the canopy is not visible, the other method must be used",
    )
    parser.add_argument(
        "--offset",
        nargs="+",
        type=float,
        default=None,
        help="Translation offset to apply to the MLS clouds",
    )
    parser.add_argument(
        "--noise_matrix",
        nargs="+",
        type=float,
        default=[
            1e06,
            0,
            0,
            0,
            0,
            0,
            1e06,
            0,
            0,
            0,
            0,
            1e06,
            0,
            0,
            0,
            10000,
            0,
            0,
            10000,
            0,
            10000,
        ],
        help="Upper triangular elements of the matrix of the 6*6 noise covariance matrix to apply to the MLS clouds",
    )
    parser.add_argument(
        "--mls_registered_cloud_folder",
        default=None,
        help="Path to the folder where the registered MLS clouds and the new pose graph will be stored",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Set this to true to visualize the intermediate steps of the pipeline",
    )
    parser.add_argument(
        "--save_pose_graph",
        default=False,
        action="store_true",
        help="Save pose graph to file",
    )
    parser.add_argument(
        "--pose_graph_file", default=None, help="Path to the MLS pose graph file"
    )
    parser.add_argument(
        "--downsample-cloud",
        default=False,
        action="store_true",
        help="Downsample input point clouds",
    )
    parser.add_argument(
        "--min_distance_between_peaks",
        type=float,
        default=2.5,
        help="Minimum distance between tree's peaks in the MLS cloud",
    )
    parser.add_argument(
        "--max_number_of_clique",
        type=int,
        default=5,
        help="Maximum number of cliques to consider for correspondence matching",
    )
    parser.add_argument(
        "--crop_mls_cloud",
        default=False,
        action="store_true",
        help="Crop the input MLS clouds",
    )
    parser.add_argument(
        "--icp_fitness_score_threshold",
        type=float,
        default=0.85,
        help="Using the ICP fitness score threshold to determine if the registration is successful. If the score is above this threshold, the registration is considered successful.",
    )
    parser.add_argument(
        "--correspondence_graph_distance_threshold",
        type=float,
        default=0.2,
        help="Maximum distance threshold to consider two edges as similar in the correspondence graph.",
    )
    parser.add_argument(
        "--maximum_rotation_offset",
        type=float,
        default=1.6,
        help="Maximum rotation offset in radians between the MLS and UAV clouds. Increasing this value will make the search space larger and the registration will take longer.\
            You can consider lowering this value if you observe that there is no rotation offset between the clouds.",
    )
    parser.add_argument("--logging_dir", type=str, help="Path of the logging directory")
    parser.add_argument(
        "--log_level",
        default="DEBUG",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
            for key, value in config.items():
                setattr(args, key, value)

    return args
