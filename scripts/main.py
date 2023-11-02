#!/usr/bin/env python3
from digiforest_registration.tasks.registration import Registration
from digiforest_registration.utils import CloudIO
from digiforest_registration.utils import crop_cloud
from pathlib import Path
import numpy as np
import os

import argparse


def crop_uav_cloud(uav_cloud, frontier_cloud):
    padding = 20
    cropped_uav_cloud = crop_cloud(uav_cloud, frontier_cloud, padding=padding)
    return cropped_uav_cloud


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser(
        prog="cloud_registration",
        description="Registers a frontier cloud to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--uav_cloud")
    parser.add_argument("--frontier_cloud", default=None)
    parser.add_argument("--frontier_cloud_folder", default=None)
    parser.add_argument("--ground_segmentation_method", nargs="?", default="default")
    parser.add_argument("--offset", nargs="+", type=float, default=None)
    parser.add_argument("--output_folder", default=None)
    args = parser.parse_args()

    # Check validity of inputs
    frontier_cloud_filenames = []
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
                    if entry.suffix == ".ply" and entry.name[:4] == "tile":
                        frontier_cloud_filenames.append(entry)

    # loading the data
    cloud_io = CloudIO(
        np.array([args.offset[0], args.offset[1], args.offset[2]], dtype=np.float32)
        if args.offset is not None and len(args.offset)
        else None
    )
    uav_cloud = cloud_io.load_cloud(str(uav_cloud_filename))
    failure_count = 0

    for frontier_cloud_filename in frontier_cloud_filenames:
        print("Processing file: ", frontier_cloud_filename.name)
        frontier_cloud = cloud_io.load_cloud(str(frontier_cloud_filename))
        cropped_uav_cloud = crop_uav_cloud(uav_cloud, frontier_cloud)

        # import open3d as o3d
        # frontier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        # cropped_uav_cloud.paint_uniform_color([0.0, 1.0, 0])
        # o3d.visualization.draw_geometries(
        #     [frontier_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
        #     window_name="Initial data",
        # )
        #

        registration = Registration(
            cropped_uav_cloud, frontier_cloud, args.ground_segmentation_method
        )
        transform, success = registration.registration()
        print("File: ", frontier_cloud_filename.name, success)
        if not success:
            failure_count += 1

        if success and args.output_folder is not None:
            output_filename = os.path.join(
                args.output_folder, frontier_cloud_filename.name
            )
            cloud_io.save_cloud(frontier_cloud, output_filename)

    print("Total number of failures: ", failure_count)
    print("Total number of clouds: ", len(frontier_cloud_filenames))
