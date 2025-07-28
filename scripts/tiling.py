import open3d as o3d
import numpy as np
from dataclasses import dataclass
import argparse
import csv
from pathlib import Path


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
    def __init__(self, tile_size: float, output_folder: str):
        self.tile_size = tile_size
        self.output_folder = Path(output_folder)

    def tile_filepath(self, row, col):
        path = self.output_folder / f"tile_{row*self.tile_size+col}.ply"
        return str(path.absolute())

    def write_point_cloud(self, filepath, cloud: o3d.t.geometry.PointCloud):
        o3d.t.io.write_point_cloud(filepath, cloud)

    def _init_grid(self, cloud: o3d.t.geometry.PointCloud):
        """Initialize the grid based on the point cloud's bounding box."""
        positions = cloud.point["positions"].numpy()
        self.x_min = np.min(positions[:, 0])
        self.y_min = np.min(positions[:, 1])
        self.x_max = np.max(positions[:, 0])
        self.y_max = np.max(positions[:, 1])
        self.num_rows = int(np.ceil((self.y_max - self.y_min) / self.tile_size))
        self.num_cols = int(np.ceil((self.x_max - self.x_min) / self.tile_size))
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

    def create_tiles(self, cloud):
        """
        Split a point cloud into tiles.
        cloud: open3d.t.geometry.PointCloud with at least 'positions'
        """
        self._init_grid(cloud)

        positions = cloud.point["positions"].numpy()  # (N, 3)
        has_colors = "colors" in cloud.point
        has_normals = "normals" in cloud.point

        if has_colors:
            colors = cloud.point["colors"].numpy()
        if has_normals:
            normals = cloud.point["normals"].numpy()

        for i in range(positions.shape[0]):
            x, y, z = positions[i]

            row = int((x - self.x_min) / self.tile_size)
            col = int((y - self.y_min) / self.tile_size)

            if row < 0 or row >= self.num_rows or col < 0 or col >= self.num_cols:
                continue

            tile = self.tile_clouds[row][col]
            if (
                x < tile.x_min
                or x > tile.x_min + tile.size_x
                or y < tile.y_min
                or y > tile.y_min + tile.size_y
            ):
                continue

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

                if len(tile.positions) == 0:
                    continue

                pc_tile = o3d.t.geometry.PointCloud()
                pc_tile.point["positions"] = o3d.core.Tensor(
                    tile.positions, dtype=o3d.core.Dtype.Float32
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
                    counter = row * self.tile_size + col
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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_inputs()
    cloud = o3d.t.io.read_point_cloud(args.cloud)
    generator = TilesGenerator(args.tile_size, args.output_folder)
    generator.create_tiles(cloud)
    generator.save_tiling_file()
    print(f"Tiling completed. Tiles saved in {args.output_folder}.")
