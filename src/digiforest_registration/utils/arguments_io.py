import yaml
import argparse


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="registration_pipeline",
        description="Registers MLS clouds to a reference UAV cloud",
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
    parser.add_argument("--optimized_cloud_output_folder", default=None)
    parser.add_argument(
        "--load_clouds", default=False, action="store_true", help="show clouds"
    )
    parser.add_argument(
        "--tiles", default=False, action="store_true", help="processing tiles"
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
        "--mls_registered_cloud_folder",
        default=None,
        help="path to the folder where the registered MLS clouds and the new pose graph will be stored",
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
