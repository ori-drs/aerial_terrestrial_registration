#!/usr/bin/env python3
from digiforest_registration.tasks.vertical_alignment import VerticalRegistration
from digiforest_registration.tasks.horizontal_alignment import HorizontalRegistration
from digiforest_registration.utils import CloudLoader
from pathlib import Path

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="cloud_registration",
        description="Registers a frontier cloud to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("uav_cloud")
    parser.add_argument("frontier_cloud")
    args = parser.parse_args()

    # Check validity of inputs
    uav_cloud_filename = Path(args.uav_cloud)
    if not uav_cloud_filename.exists():
        raise ValueError(f"Input file [{uav_cloud_filename}] does not exist")

    frontier_cloud_filename = Path(args.frontier_cloud)
    if not frontier_cloud_filename.exists():
        raise ValueError(f"Input file [{frontier_cloud_filename}] does not exist")

    loader = CloudLoader()

    uav_cloud = loader.load_cloud(str(uav_cloud_filename))
    frontier_cloud = loader.load_cloud(str(frontier_cloud_filename))

    vertical_registration = VerticalRegistration(uav_cloud, frontier_cloud)
    # (uav_groud_plane, frontier_ground_plane) = vertical_registration.process()

    # results of the vertical registration, use them directly to save time
    uav_groud_plane = [
        -0.028127017612793577,
        0.03713812538079706,
        0.9989142258089079,
        19.434460578098562,
    ]
    frontier_ground_plane = [
        -0.027477370673811043,
        0.0419955118046868,
        0.9987398916079784,
        9.787417210868085,
    ]

    horizontal_registration = HorizontalRegistration(
        uav_cloud, uav_groud_plane, frontier_cloud, frontier_ground_plane
    )
    horizontal_registration.process()
