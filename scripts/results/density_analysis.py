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
    parser.add_argument("--cloud")
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
            for key, value in config.items():
                setattr(args, key, value)

    return args


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
    cloud = cloud_io.load_cloud(args.cloud)

    # compute density
    density = [0]
    step = 0.5
    bbox = cloud.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound.numpy()
    max_bound = bbox.max_bound.numpy()

    points = cloud.point.positions.numpy()
    x = np.arange(min_bound[0], max_bound[0], step)
    z = np.arange(0, max_bound[2], step)
    sorted_indices = np.argsort(points[:, 2])
    sorted_points = points[sorted_indices]

    for i in range(len(z) - 1):
        mask = (sorted_points[:, 2] > z[i]) & (sorted_points[:, 2] < z[i + 1])
        density.append(np.sum(mask))

    # Display density
    plt.plot(z, density, label="Density")

    # Add labels and title
    plt.xlabel("Z axis (m)")
    plt.ylabel("Point Density")
    plt.title("Density Plot")

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()
