#!/usr/bin/env python3
from digiforest_registration.tasks.registration import Registration
from digiforest_registration.utils import CloudLoader
from pathlib import Path
import numpy as np

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
    args = parser.parse_args()

    # Check validity of inputs
    uav_cloud_filename = Path(args.uav_cloud)
    if not uav_cloud_filename.exists():
        raise ValueError(f"Input file [{uav_cloud_filename}] does not exist")

    frontier_cloud_filename = Path(args.frontier_cloud)
    if not frontier_cloud_filename.exists():
        raise ValueError(f"Input file [{frontier_cloud_filename}] does not exist")

    # loading the data
    loader = CloudLoader(args.offset)
    uav_cloud = loader.load_cloud(str(uav_cloud_filename))
    frontier_cloud = loader.load_cloud(str(frontier_cloud_filename))

    registration = Registration(
        uav_cloud, frontier_cloud, args.ground_segmentation_method
    )
    success = registration.registration()
