#!/usr/bin/env python3
from digiforest_registration.tasks.registration import Registration
from digiforest_registration.utils import CloudIO
from pathlib import Path
import numpy as np
import os

import argparse

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser(
        prog="cloud_registration",
        description="Registers a frontier cloud to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--uav_cloud")
    parser.add_argument("--frontier_cloud")
    parser.add_argument("--ground_segmentation_method", nargs="?", default="default")
    parser.add_argument("--offset", nargs="+", type=float, default=None)
    parser.add_argument("--output_folder", default=None)
    args = parser.parse_args()

    # Check validity of inputs
    uav_cloud_filename = Path(args.uav_cloud)
    if not uav_cloud_filename.exists():
        raise ValueError(f"Input file [{uav_cloud_filename}] does not exist")

    frontier_cloud_filename = Path(args.frontier_cloud)
    if not frontier_cloud_filename.exists():
        raise ValueError(f"Input file [{frontier_cloud_filename}] does not exist")

    # loading the data
    cloud_io = CloudIO(
        np.array([args.offset[0], args.offset[1], args.offset[2]], dtype=np.float32)
        if args.offset is not None and len(args.offset)
        else None
    )
    uav_cloud = cloud_io.load_cloud(str(uav_cloud_filename))
    frontier_cloud = cloud_io.load_cloud(str(frontier_cloud_filename))

    registration = Registration(
        uav_cloud, frontier_cloud, args.ground_segmentation_method
    )
    transform, success = registration.registration()

    if success and args.output_folder is not None:
        output_filename = os.path.join(args.output_folder, frontier_cloud_filename.name)
        cloud_io.save_cloud(frontier_cloud, output_filename)
