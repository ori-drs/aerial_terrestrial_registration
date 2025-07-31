import open3d as o3d
import numpy as np
from dataclasses import dataclass
import argparse
import csv
from pathlib import Path

from digiforest_registration.utils import CloudIO


@dataclass
class Tile:
    x_min: float
    y_min: float
    size_x: float
    size_y: float
    positions: list
    colors: list = None
    normals: list = None


class TilesGenerator:
    def __init__(self, tile_size: float, output_folder: str, cloud_io: CloudIO):
        self.tile_size = tile_size
        self.output_folder = Path(output_folder)
        self.cloud_io = cloud_io

    def _tile_count(self, row, col):
        return row * self.num_cols + col

    def tile_filepath(self, row, col):
        path = self.output_folder / f"tile_{self._tile_count(row, col)}.ply"
        return str(path.absolute())

    def write_point_cloud(self, filepath, cloud: o3d.t.geometry.PointCloud):
        self.cloud_io.save_cloud(cloud, filepath, local_coordinates=False)

    def _init_grid(self, cloud: o3d.t.geometry.PointCloud):
        """Initialize the grid based on the point cloud's bounding box."""
        positions = cloud.point["positions"].numpy()
        self.x_min = np.min(positions[:, 0])
        self.y_min = np.min(positions[:, 1])
        self.x_max = np.max(positions[:, 0])
        self.y_max = np.max(positions[:, 1])
        self.num_rows = int(np.ceil((self.y_max - self.y_min) / self.tile_size))
        self.num_cols = int(np.ceil((self.x_max - self.x_min) / self.tile_size))
        print(f"Grid initialized: {self.num_rows} rows, {self.num_cols} cols")
        self.tile_clouds = [
            [None for _ in range(self.num_cols)] for _ in range(self.num_rows)
        ]

        x = self.x_min
        for row in range(0, self.num_rows):
            y = self.y_min
            for col in range(0, self.num_cols):
                self.tile_clouds[row][col] = Tile(
                    x_min=x,
                    y_min=y,
                    size_x=self.tile_size,
                    size_y=self.tile_size,
                    positions=[],
                    colors=[],
                    normals=[],
                )
                y += self.tile_size
            x += self.tile_size

    def create_tiles(self, filename: str):
        """
        Split a point cloud into tiles.
        filename: file path to the point cloud.
        """
        cloud = self.cloud_io.load_cloud(filename)
        self._init_grid(cloud)

        positions = cloud.point["positions"].numpy()  # (N, 3)
        has_colors = "colors" in cloud.point
        has_normals = "normals" in cloud.point

        if has_colors:
            colors = cloud.point["colors"].numpy()
        if has_normals:
            normals = cloud.point["normals"].numpy()

        for i in range(positions.shape[0]):
            if i % (positions.shape[0] // 10) == 0 and i > 0:
                print(f"Processing cloud: {(100*i)//positions.shape[0]}% done")
            x, y, z = positions[i]

            row = int((y - self.y_min) / self.tile_size)
            col = int((x - self.x_min) / self.tile_size)

            if row < 0 or row >= self.num_rows or col < 0 or col >= self.num_cols:
                continue

            tile = self.tile_clouds[row][col]

            tile.positions.append([x, y, z])
            if has_colors:
                tile.colors.append(colors[i])
            if has_normals:
                tile.normals.append(normals[i])

        # Save each tile cloud
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                tile = self.tile_clouds[row][col]
                if tile is None:
                    continue

                pc_tile = o3d.t.geometry.PointCloud()
                pc_tile.point["positions"] = o3d.core.Tensor(
                    tile.positions, dtype=o3d.core.Dtype.Float64
                )

                if has_colors:
                    pc_tile.point["colors"] = o3d.core.Tensor(
                        tile.colors, dtype=o3d.core.Dtype.Float32
                    )
                if has_normals:
                    pc_tile.point["normals"] = o3d.core.Tensor(
                        tile.normals, dtype=o3d.core.Dtype.Float32
                    )

                filepath = self.tile_filepath(row, col)
                self.write_point_cloud(filepath, pc_tile)

    def save_tiling_file(self):
        tiles_filename = self.output_folder / "tiles.csv"
        with open(tiles_filename, mode="w", newline="") as tiles_file:
            writer = csv.writer(tiles_file)
            writer.writerow(["#counter", "x_min", "y_min", "size_x", "size_y"])
            for row in range(len(self.tile_clouds)):
                for col in range(len(self.tile_clouds[row])):
                    tile = self.tile_clouds[row][col]

                    if tile is None:
                        continue

                    counter = self._tile_count(row, col)
                    writer.writerow(
                        [
                            counter,
                            f"{tile.x_min:.4f}",
                            f"{tile.y_min:.4f}",
                            f"{tile.size_x:.1f}",
                            f"{tile.size_y:.1f}",
                        ]
                    )


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="Tiling",
        description="Cut a point cloud into tiles of a given size.",
    )
    parser.add_argument("--cloud", help="the path to the input point cloud")
    parser.add_argument(
        "--tile_size",
        type=int,
        default=20,
        help="the size of the tiles to create in meters",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        help="the path to the folder containing the MLS clouds. Set either this parameter or mls_cloud",
    )
    parser.add_argument(
        "--offset",
        nargs="+",
        type=float,
        default=None,
        help="translation offset to apply to the MLS clouds",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_inputs()
    cloud = o3d.t.io.read_point_cloud(args.cloud)
    offset = None
    if args.offset is not None and len(args.offset) == 3:
        offset = np.array(
            [args.offset[0], args.offset[1], args.offset[2]], dtype=np.float32
        )

    cloud_io = CloudIO(offset)
    generator = TilesGenerator(args.tile_size, args.output_folder, cloud_io)
    generator.create_tiles(args.cloud)
    generator.save_tiling_file()
    print(f"Tiling completed. Tiles saved in {args.output_folder}.")
