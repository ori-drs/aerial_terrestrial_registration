import open3d as o3d
import numpy as np


class CloudLoader:
    def __init__(self, offset):
        self.offset = offset

    def load_cloud(self, filename):
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
                cloud = cloud.translate(-point)
                self.offset = -point

        return cloud
