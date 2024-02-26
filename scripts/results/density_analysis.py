#!/usr/bin/env python3
from digiforest_registration.utils import CloudIO
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import argparse
import yaml


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="cloud_registration",
        description="Registers a frontier cloud to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--config", default=None, help="yaml config file")
    parser.add_argument("--offset", nargs="+", type=float, default=None)
    parser.add_argument("--uav_cloud", default=None)
    parser.add_argument("--frontier_cloud", default=None)
    parser.add_argument("--combined_cloud", default=None)
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
            for key, value in config.items():
                setattr(args, key, value)

    return args


def compute_density(cloud, z):
    density = [0]

    points = cloud.point.positions.numpy()
    sorted_indices = np.argsort(points[:, 2])
    sorted_points = points[sorted_indices]

    for i in range(len(z) - 1):
        mask = (sorted_points[:, 2] > z[i]) & (sorted_points[:, 2] < z[i + 1])
        density.append(np.sum(mask))

    return density


def compute_axis(clouds, step):
    max_z = 0
    for cloud in clouds:
        bbox = cloud.get_axis_aligned_bounding_box()
        max_bound = bbox.max_bound.numpy()
        max_z = max(max_bound[2], max_z)

    z = np.arange(0, max_z, step)
    return z


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # set seed for deterministic results
    o3d.utility.random.seed(12345)

    args = parse_inputs()

    # Loading the data
    offset = None
    if args.offset is not None and len(args.offset) == 3:
        offset = np.array(
            [args.offset[0], args.offset[1], args.offset[2]], dtype=np.float32
        )

    cloud_io = CloudIO(offset)
    uav_cloud = cloud_io.load_cloud(args.uav_cloud)
    frontier_cloud = cloud_io.load_cloud(args.frontier_cloud)
    combined_cloud = cloud_io.load_cloud(args.combined_cloud)

    # compute density
    step = 0.5
    # z = compute_axis([uav_cloud, frontier_cloud, combined_cloud], step)
    z = np.arange(0, 20, step)
    density_uav = compute_density(uav_cloud, z)
    density_frontier = compute_density(frontier_cloud, z)
    density_combined = compute_density(combined_cloud, z)

    # Display density
    plt.plot(z, density_uav, label="Aerial cloud", color="#5f8dd3")
    plt.plot(z, density_frontier, label="Terrestrial cloud", color="#f1943b")
    plt.plot(z, density_combined, label="Combined cloud", color="#22de0d")
    plt.yticks([])

    # Add labels and title
    plt.xlabel("Z axis (m)")
    plt.ylabel("Point Density")
    plt.title("Density Plot")

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()
