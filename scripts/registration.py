#!/usr/bin/env python3
from digiforest_registration.tasks.vertical_alignment import VerticalRegistration
from digiforest_registration.tasks.horizontal_alignment import HorizontalRegistration
from digiforest_registration.utils import CloudLoader
from digiforest_registration.utils import euler_to_rotation_matrix
from pathlib import Path

import argparse
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="cloud_registration",
        description="Registers a frontier cloud to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("uav_cloud")
    parser.add_argument("frontier_cloud")
    parser.add_argument("ground_segmentation_method", nargs="?", default="default")
    args = parser.parse_args()

    # Check validity of inputs
    uav_cloud_filename = Path(args.uav_cloud)
    if not uav_cloud_filename.exists():
        raise ValueError(f"Input file [{uav_cloud_filename}] does not exist")

    frontier_cloud_filename = Path(args.frontier_cloud)
    if not frontier_cloud_filename.exists():
        raise ValueError(f"Input file [{frontier_cloud_filename}] does not exist")

    # loading the data
    loader = CloudLoader()
    uav_cloud = loader.load_cloud(str(uav_cloud_filename))
    frontier_cloud = loader.load_cloud(str(frontier_cloud_filename))

    # transformation matrix from frontier cloud to uav cloud that we are estimating
    transform = np.identity(4)

    vertical_registration = VerticalRegistration(
        uav_cloud,
        frontier_cloud,
        ground_segmentation_method=args.ground_segmentation_method,
    )
    (uav_groud_plane, frontier_ground_plane, tz) = vertical_registration.process()
    transform[2, 3] = tz

    # results of the vertical registration, use them directly to save time
    # uav_groud_plane = [
    #     -0.028127017612793577,
    #     0.03713812538079706,
    #     0.9989142258089079,
    #     19.434460578098562,
    # ]
    # frontier_ground_plane = [
    #     -0.027477370673811043,
    #     0.0419955118046868,
    #     0.9987398916079784,
    #     9.787417210868085,
    # ]

    horizontal_registration = HorizontalRegistration(
        uav_cloud, uav_groud_plane, frontier_cloud, frontier_ground_plane
    )
    (tx, ty, yaw) = horizontal_registration.process()

    R = euler_to_rotation_matrix(yaw, 0, 0)
    transform[0:3, 0:3] = R
    transform[0, 3] = tx
    transform[1, 3] = ty

    print("Transformation matrix:")
    print(transform)

    # Visualize the results
    frontier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    uav_cloud.paint_uniform_color([0.0, 1.0, 0])
    o3d.visualization.draw_geometries(
        [frontier_cloud.to_legacy(), uav_cloud.to_legacy()]
    )

    frontier_cloud.transform(transform)

    o3d.visualization.draw_geometries(
        [frontier_cloud.to_legacy(), uav_cloud.to_legacy()]
    )
