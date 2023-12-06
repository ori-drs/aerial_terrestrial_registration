# Author: Matias Mattamala (matias@robots.ox.ac.uk)

import numpy as np
import gtsam
from pathlib import Path
from digiforest_registration.optimization.pose_graph import PoseGraph
from digiforest_registration.utils import rotation_matrix_to_quat, get_cloud_center


def read_pose_gt(tokens):
    sec = tokens[0]
    nsec = tokens[1]
    pos = [float(i) for i in tokens[2:5]]
    q = [float(i) for i in tokens[5:9]]
    quat = gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2])
    pose = gtsam.Pose3(quat, pos)

    return pose, (sec, nsec)


def read_pose_slam(tokens):
    pose_id = int(tokens[0])
    pose_stamp = (tokens[8], tokens[9])
    pos = [float(i) for i in tokens[1:4]]
    q = [float(i) for i in tokens[4:8]]
    quat = gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2])
    pose = gtsam.Pose3(quat, pos)

    return pose, pose_stamp, pose_id


def read_pose_edge_slam(tokens):
    parent_id = int(tokens[0])
    child_id = int(tokens[1])
    pos = [float(i) for i in tokens[2:5]]
    q = [float(i) for i in tokens[5:9]]
    quat = gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2])
    relative_pose = gtsam.Pose3(quat, pos)
    upper_triangular = [float(i) for i in tokens[9:]]
    relative_info = np.eye(6, dtype=np.float64)
    relative_info[0, 0:6] = relative_info[0:6, 0] = upper_triangular[0:6]
    relative_info[1, 1:6] = relative_info[1:6, 1] = upper_triangular[6:11]
    relative_info[2, 2:6] = relative_info[2:6, 2] = upper_triangular[11:15]
    relative_info[3, 3:6] = relative_info[3:6, 3] = upper_triangular[15:18]
    relative_info[4, 4:6] = relative_info[4:6, 4] = upper_triangular[18:20]
    relative_info[5, 5:6] = relative_info[5:6, 5] = upper_triangular[20:21]
    # the convention is translation, rotation, which is the opposite of gtsam
    # swapping blocks to match gtsam convention
    relative_info_copy = relative_info.copy()
    relative_info[0:3, 0:3] = relative_info_copy[3:6, 3:6]
    relative_info[0:3, 3:6] = relative_info_copy[3:6, 0:3]
    relative_info[3:6, 0:3] = relative_info_copy[0:3, 3:6]
    relative_info[3:6, 3:6] = relative_info_copy[0:3, 0:3]

    return relative_pose, relative_info, parent_id, child_id


def load_pose_graph(path: str, clouds_folder_path, cloud_loader):
    graph = PoseGraph()
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.strip().split(" ")

            # Parse lines
            if tokens[0] == "#":
                continue

            elif tokens[0] == "PLATFORM_ID":
                continue

            elif tokens[0] == "VERTEX_SE3:QUAT_TIME":
                pose, pose_stamp, pose_id = read_pose_slam(tokens[1:])
                graph.add_node(pose_id, pose_stamp, pose)

                # if clouds_path is not None:
                #     cloud = read_cloud(pose_stamp, clouds_path)
                #     cloud.transform(T_base_lidar)
                #     graph.add_clouds(pose_id, cloud)

            elif tokens[0] == "EDGE_SE3:QUAT":
                relative_pose, relative_info, parent_id, child_id = read_pose_edge_slam(
                    tokens[1:]
                )
                if parent_id != child_id:
                    graph.add_edge(
                        parent_id, child_id, "center", relative_pose, relative_info
                    )
                else:
                    # Add transform between the node and the aerial cloud
                    # get center of the tile
                    tile = "tile_" + str(parent_id) + ".ply"
                    cloud_path = clouds_folder_path / tile
                    cloud = cloud_loader.load_cloud(str(cloud_path))
                    quat = gtsam.Rot3.Quaternion(1, 0, 0, 0)
                    center_pose = gtsam.Pose3(quat, get_cloud_center(cloud))
                    # add the edge
                    uav_pose = center_pose.inverse() * relative_pose
                    graph.add_edge(
                        parent_id, child_id, "aerial", uav_pose, relative_info
                    )

    return graph


def write_graph_node(node):
    id = node["id"]
    sec, nsec = node["stamp"]
    x = node["pose"].translation()[0]
    y = node["pose"].translation()[1]
    z = node["pose"].translation()[2]

    qx = node["pose"].rotation().toQuaternion().x()
    qy = node["pose"].rotation().toQuaternion().y()
    qz = node["pose"].rotation().toQuaternion().z()
    qw = node["pose"].rotation().toQuaternion().w()

    stream = f"VERTEX_SE3:QUAT_TIME {id} {x} {y} {z} {qx} {qy} {qz} {qw} {sec} {nsec}\n"
    return stream


def write_graph_edge(edge):
    parent_id = edge["parent_id"]
    child_id = edge["child_id"]

    x = edge["pose"].translation()[0]
    y = edge["pose"].translation()[1]
    z = edge["pose"].translation()[2]

    qx = edge["pose"].rotation().toQuaternion().x()
    qy = edge["pose"].rotation().toQuaternion().y()
    qz = edge["pose"].rotation().toQuaternion().z()
    qw = edge["pose"].rotation().toQuaternion().w()

    info = ""
    for i in range(6):
        for j in range(i, 6):
            info += f"{edge['info'][i][j]} "
    info = info[:-1]

    stream = (
        f"EDGE_SE3:QUAT {parent_id} {child_id} {x} {y} {z} {qx} {qy} {qz} {qw} {info}\n"
    )
    return stream


def write_pose_graph(pose_graph: PoseGraph, path: str):
    with open(path, "w") as file:
        for node in pose_graph.nodes:
            file.write(write_graph_node(node))
        for edge in pose_graph.edges:
            file.write(write_graph_edge(edge))


def write_tiles_to_pose_graph_file(
    tiles_folder_path: str,
    pose_graph_path: str,
    grid_size_row: int,
    grid_size_col: int,
    registration_results: dict,
    cloud_loader,
):
    def get_tile_number(filename):
        # assuming a filename of the form: tile_0000.ply
        underscore_index = filename.find("_")
        dot_index = filename.find(".")
        return int(filename[underscore_index + 1 : dot_index])

    # the tiles form a grid, building the grid
    folder = Path(tiles_folder_path)
    cloud_paths = []
    if not folder.is_dir():
        raise ValueError(f"Input folder [{folder}] does not exist")
    else:
        # Get all the ply files in the folder
        for entry in folder.iterdir():
            if entry.is_file():
                if entry.suffix == ".ply":
                    cloud_paths.append(entry)

    if grid_size_row * grid_size_col != len(cloud_paths):
        raise ValueError(
            f"Input folder [{folder}] does not contain an unexpected number of tiles"
        )

    coordinates = []  # (cloud_path, center)
    for cloud_path in cloud_paths:
        cloud = cloud_loader.load_cloud(str(cloud_path))
        center = get_cloud_center(cloud)
        coordinates.append((cloud_path, center))

    # sort the x coordinates
    coordinates.sort(key=lambda x: x[1][0])  # column major
    for col in range(grid_size_col):
        # sort the y coordinates
        coordinates[col * grid_size_row : (col + 1) * grid_size_row] = sorted(
            coordinates[col * grid_size_row : (col + 1) * grid_size_row],
            key=lambda x: x[1][1],
        )

    # write the pose graph
    with open(pose_graph_path, "w") as file:
        for i in range(len(coordinates)):
            filename = coordinates[i][0].name
            tile_id = get_tile_number(filename)
            if not registration_results[filename].success:
                continue

            # write the node
            center = coordinates[i][1]
            file.write(
                f"VERTEX_SE3:QUAT_TIME {tile_id} {center[0]:.2f} {center[1]:.2f} {center[2]:.2f} 0 0 0 1 0 0\n"
            )

        saved_edges = []
        for i in range(len(coordinates)):
            filename = coordinates[i][0].name
            tile_id = get_tile_number(filename)
            # write the edge
            # 4 neighbours
            col = i // grid_size_row
            row = i % grid_size_row

            neighbours_row = [-1, 0, 0, 1]
            neighbours_col = [0, -1, 1, 0]
            for j in range(4):
                neighbour_row = row + neighbours_row[j]
                neighbour_col = col + neighbours_col[j]
                if (
                    neighbour_row >= 0
                    and neighbour_row < grid_size_row
                    and neighbour_col >= 0
                    and neighbour_col < grid_size_col
                ):
                    # TODO remove extra edges
                    neighbour = neighbour_col * grid_size_row + neighbour_row
                    neighbour_id = get_tile_number(coordinates[neighbour][0].name)
                    if not registration_results[coordinates[neighbour][0].name].success:
                        continue

                    if (tile_id, neighbour_id) in saved_edges or (
                        neighbour_id,
                        tile_id,
                    ) in saved_edges:
                        continue

                    center = coordinates[i][1]  # tail
                    center_neighbour = coordinates[neighbour][1]  # head
                    offset = center - center_neighbour

                    # write the edge
                    file.write(
                        f"EDGE_SE3:QUAT {tile_id} {neighbour_id} {offset[0]:.2f} {offset[1]:.2f} {offset[2]:.2f} 0 0 0 1 1e+06 0 0 0 0 0 1e+06 0 0 0 0 1e+06 0 0 0 10000 0 0 10000 0 10000\n"
                    )
                    saved_edges.append((tile_id, neighbour_id))

            # write the extra edge, between tile and aerial cloud
            mat = registration_results[filename].transform
            quat = rotation_matrix_to_quat(mat[0:3, 0:3])
            file.write(
                f"EDGE_SE3:QUAT {tile_id} {tile_id} {mat[0, 3]:.2f} {mat[1, 3]:.2f} {mat[2, 3]:.2f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f} {quat[0]} 1e+06 0 0 0 0 0 1e+06 0 0 0 0 1e+06 0 0 0 10000 0 0 10000 0 10000\n"
            )
