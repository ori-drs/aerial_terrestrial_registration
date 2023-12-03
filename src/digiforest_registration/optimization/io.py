# Author: Matias Mattamala (matias@robots.ox.ac.uk)

import numpy as np
import gtsam
from pathlib import Path
from digiforest_registration.optimization.pose_graph import PoseGraph
from digiforest_registration.utils import rotation_matrix_to_quat


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

    return relative_pose, relative_info, parent_id, child_id


def load_pose_graph(path: str, clouds_path: str = None):
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
                edge_type = "odometry" if (parent_id == child_id - 1) else "loop"
                graph.add_edge(
                    parent_id, child_id, edge_type, relative_pose, relative_info
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


# TODO move this function elsewhere
def write_tiles_to_pose_graph_file(
    tiles_folder_path: str,
    pose_graph_path: str,
    registration_results: dict,
    cloud_loader,
):

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

    grid_size = int(np.sqrt(len(cloud_paths)))

    if grid_size * grid_size != len(cloud_paths):
        raise ValueError(
            f"Input folder [{folder}] does not contain a square number of tiles"
        )

    grid = [str() for i in range(grid_size) for j in range(grid_size)]
    coordinates = []
    for cloud_path in cloud_paths:
        frontier_cloud = cloud_loader.load_cloud(str(cloud_path))
        # get bounding box
        bbox = frontier_cloud.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound.numpy()
        max_bound = bbox.max_bound.numpy()
        # get center of cloud
        center = (min_bound + max_bound) / 2
        coordinates.append((cloud_path, center))

    # sort the x coordinates
    coordinates.sort(key=lambda x: x[1][0])
    for i in range(grid_size):
        # sort the y coordinates
        coordinates[i * grid_size : (i + 1) * grid_size].sort(key=lambda x: x[1][1])
        # fill the grid
        for j in range(grid_size):
            grid[i * grid_size + j] = coordinates[i * grid_size + j][0]

    # write the pose graph
    with open(pose_graph_path, "w") as file:
        for i in range(len(coordinates)):
            if not registration_results[coordinates[i][0]].success:
                continue

            # write the node
            file.write(
                f"VERTEX_SE3:QUAT_TIME {i} {coordinates[i][1][0]} {coordinates[i][1][1]} {coordinates[i][1][2]} 0 0 0 1 0 0\n"
            )

        # write the edge
        # 4 neighbours
        row = i // grid_size
        col = i % grid_size
        neighbours_row = [-1, 0, 0, 1]
        neighbours_col = [0, -1, 1, 0]
        for j in range(4):
            neighbour_row = row + neighbours_row[j]
            neighbour_col = col + neighbours_col[j]
            if (
                neighbour_row >= 0
                and neighbour_row < grid_size
                and neighbour_col >= 0
                and neighbour_col < grid_size
            ):
                neighbour = neighbour_row * grid_size + neighbour_col
                if not registration_results[coordinates[neighbour][0]].success:
                    continue
                file.write(
                    f"EDGE_SE3:QUAT {i} {neighbour} 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0\n"
                )
            # write the extra edge
            mat = registration_results[coordinates[i][0]].transform
            quat = rotation_matrix_to_quat(mat[0:3, 0:3])
            file.write(
                f"EDGE_SE3:QUAT {i} {i} {mat[0, 3]} {mat[1, 3]} {mat[2, 3]} {quat[1]} {quat[2]} {quat[3]} {quat[0]} 0 0 0 0 0 0 0 0 0 0\n"
            )
