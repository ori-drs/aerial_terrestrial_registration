import open3d as o3d
import numpy as np


class CloudIO:
    def __init__(self, offset: np.ndarray):
        self.offset = offset  # to transform cloud to local coordinates

    def load_cloud(self, filename: str):
        """
        Loads a point cloud from a file and translates it if its coordinates are too large."""

        cloud = o3d.t.io.read_point_cloud(filename)
        threshold = 10**6
        if self.offset is not None:
            cloud = cloud.translate(self.offset)
        elif len(cloud.point.positions) > 0:
            point = cloud.point.positions[0].numpy().copy()
            if (
                (np.abs(point[0]) > threshold)
                or (np.abs(point[1]) > threshold)
                or (np.abs(point[2]) > threshold)
            ):
                self.offset = -point
                print("Offset", self.offset)
                cloud = cloud.translate(self.offset)

        return cloud

    def save_cloud(self, cloud, filename: str, local_coordinates=True):
        """
        Saves a point cloud to a file."""
        if local_coordinates:
            o3d.t.io.write_point_cloud(filename, cloud)
        else:
            utm_cloud = cloud.clone()
            utm_cloud = utm_cloud.translate(-self.offset)
            o3d.t.io.write_point_cloud(filename, utm_cloud)
