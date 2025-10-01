import open3d as o3d
import numpy as np
from pathlib import Path
import logging


def is_cloud_name(path: Path):
    """
    Returns True if the path is a valid cloud file name."""
    return path.suffix == ".ply" and (
        path.name[:4] == "tile" or path.name[:5] == "cloud"
    )


def get_payload_cloud_timestamp(path: Path):
    """
    Returns the timestamp of a payload cloud file name."""
    name = path.stem
    x = name.split("_")
    return (x[1], x[2])


def get_tile_id(filename: str):
    # assuming a filename of the form: tile_0000.ply
    underscore_index = filename.find("_")
    dot_index = filename.find(".")
    return int(filename[underscore_index + 1 : dot_index])


def get_tile_filename(id: int):
    # assuming a filename of the form: tile_id.ply
    filename = "tile_" + str(id) + ".ply"
    return filename


class CloudIO:
    def __init__(self, offset: np.ndarray, logger=None, downsample_cloud=False):
        self.offset = offset  # to transform cloud to local coordinates
        self.logger = logger or logging.getLogger(
            __name__ + ".null"
        )  # no-op logger if logger is not provided
        self.downsample_cloud = downsample_cloud

    def load_cloud(self, filename: str):
        """
        Loads a point cloud from a file and translates it if its coordinates are too large."""

        cloud = o3d.t.io.read_point_cloud(filename)
        self.logger.debug(f"Loaded cloud with {len(cloud.point.positions)} points")
        if self.downsample_cloud:
            # TODO there can be a problem if the voxel_size is greater than the resolution of the canopy height image
            cloud = cloud.voxel_down_sample(voxel_size=0.08)
            self.logger.debug(f"{len(cloud.point.positions)} points remaining")

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
                cloud = cloud.translate(self.offset)

        return cloud

    def save_cloud(self, cloud, filename: str, local_coordinates=True):
        """Save a point cloud to a file.
        cloud: open3d.t.geometry.PointCloud
        filename: file path to save the point cloud.
        local_coordinates: if True, the cloud is saved in local coordinates (i.e., translated by the offset).
        If False, the cloud is saved in global coordinates.
        """
        if local_coordinates or self.offset is None:
            o3d.t.io.write_point_cloud(filename, cloud)
        else:
            cloud_transformed = cloud.clone()
            cloud_transformed = cloud_transformed.translate(-self.offset)
            o3d.t.io.write_point_cloud(filename, cloud_transformed)


class TileConfigReader:
    """Reads tile configuration from a file."""

    def __init__(self, path: Path, offset: np.ndarray):
        self.path = path
        self.offset = offset
        self.num_grid_cols = 0
        self.num_grid_rows = 0

        coordinates = []
        with open(path, "r") as file:
            lines = file.readlines()
            for line in lines:
                tokens = line.strip().split(",")

                # Parse lines
                if tokens[0][0] == "#":
                    continue

                # counter,x_min,y_min,size_x,size_y
                self.num_grid_rows = max(int(tokens[1]) + 1, self.num_grid_rows)
                self.num_grid_cols = max(int(tokens[2]) + 1, self.num_grid_cols)
                tile_id = int(tokens[0])
                x_min = float(tokens[3])
                y_min = float(tokens[4])
                size_x = float(tokens[5])
                size_y = float(tokens[6])

                center = np.array([x_min + size_x / 2, y_min + size_y / 2, 0])
                # apply offset
                if self.offset is not None:
                    center = center + self.offset

                coordinates.append((tile_id, center))

        self.coordinates = coordinates

    def get_tiles_coordinates(self, tiles_folder: Path):
        coordinates = []
        cloud_paths = []
        if not tiles_folder.is_dir():
            raise ValueError(f"Input folder {str(tiles_folder)} does not exist")
        else:
            # Get all the ply files in the folder
            for entry in tiles_folder.iterdir():
                if entry.is_file():
                    if is_cloud_name(entry):
                        cloud_paths.append(entry)

        for cloud_path in cloud_paths:
            tile_id = get_tile_id(cloud_path.name)
            for i in range(len(self.coordinates)):
                if self.coordinates[i][0] == tile_id:
                    coordinates.append(self.coordinates[i])
                    break

        # sort the x coordinates
        coordinates.sort(key=lambda x: x[1][0])  # column major
        for col in range(self.num_grid_cols):
            # sort the y coordinates
            coordinates[
                col * self.num_grid_rows : (col + 1) * self.num_grid_rows
            ] = sorted(
                coordinates[col * self.num_grid_rows : (col + 1) * self.num_grid_rows],
                key=lambda x: x[1][1],
            )

        return coordinates
