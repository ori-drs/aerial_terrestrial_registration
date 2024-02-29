#!/usr/bin/env python3
from digiforest_registration.utils import CloudIO
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import argparse
import pickle
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
    parser.add_argument("--uav_data", default=None)
    parser.add_argument("--frontier_data", default=None)
    parser.add_argument("--combined_data", default=None)
    parser.add_argument("--plot", default=False, action="store_true")
    parser.add_argument("--load_data", default=False, action="store_true")
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
            for key, value in config.items():
                setattr(args, key, value)

    return args


def compute_density(cloud, z):
    density = [0]

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        cloud.to_legacy(), voxel_size=0.05
    )
    list_points = [
        voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        for voxel in voxel_grid.get_voxels()
    ]
    points = np.array(list_points)

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


def plot(z, density_uav, density_frontier, density_combined):
    plt.plot(z[1:], density_uav[1:], label="Aerial cloud", color="#5f8dd3")
    plt.plot(z[1:], density_frontier[1:], label="Terrestrial cloud", color="#f1943b")
    plt.plot(z[1:], density_combined[1:], label="Combined cloud", color="#0f9156")

    # Add labels and title
    plt.xlabel("Z axis (m)")
    plt.ylabel("Occupied voxels")
    plt.title("Density Plot")

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # set seed for deterministic results
    o3d.utility.random.seed(12345)

    args = parse_inputs()

    step = 0.5
    # z = compute_axis([uav_cloud, frontier_cloud, combined_cloud], step)
    z = np.arange(0, 20, step)

    if args.load_data:
        file = open(args.uav_data, "rb")
        density_uav = pickle.load(file)
        file.close()
        file = open(args.frontier_data, "rb")
        density_frontier = pickle.load(file)
        file.close()
        file = open(args.combined_data, "rb")
        density_combined = pickle.load(file)
        file.close()
        plot(z, density_uav, density_frontier, density_combined)
    else:

        # Loading the clouds
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
        print("Computing density for uav cloud")
        density_uav = compute_density(uav_cloud, z)
        print("Computing density for frontier cloud")
        density_frontier = compute_density(frontier_cloud, z)
        print("Computing density for combined cloud")
        density_combined = compute_density(combined_cloud, z)

        if args.plot:
            plot(z, density_uav, density_frontier, density_combined)
        else:
            file = open("uav_data.pkl", "wb")
            pickle.dump(density_uav, file)
            file.close()
            file = open("frontier_data.pkl", "wb")
            pickle.dump(density_frontier, file)
            file.close()
            file = open("combined_data.pkl", "wb")
            pickle.dump(density_combined, file)
            file.close()
