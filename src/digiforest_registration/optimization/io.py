import numpy as np
import gtsam
from pathlib import Path
from digiforest_registration.optimization.pose_graph import PoseGraph
from digiforest_registration.utils import rotation_matrix_to_quat, get_tile_filename


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


def load_pose_graph(path: str, clouds_folder_path: Path, cloud_loader):
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

                if clouds_folder_path is not None:
                    tile_name = "tile_" + str(pose_id) + ".ply"
                    cloud_path = clouds_folder_path / tile_name
                    cloud = cloud_loader.load_cloud(str(cloud_path))

                    graph.add_clouds(pose_id, cloud, tile_name)

            elif tokens[0] == "EDGE_SE3:QUAT":
                relative_pose, relative_info, parent_id, child_id = read_pose_edge_slam(
                    tokens[1:]
                )
                if parent_id != child_id:
                    graph.add_edge(
                        parent_id, child_id, "center", relative_pose, relative_info
                    )
                else:
                    graph.add_edge(
                        parent_id, child_id, "aerial", relative_pose, relative_info
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
    tiles_folder_path: Path,
    pose_graph_path: str,
    grid_size_row: int,
    grid_size_col: int,
    registration_results: dict,
    tiles_config_reader,
):

    coordinates = tiles_config_reader.get_tiles_coordinates(tiles_folder_path)

    # write the pose graph
    with open(pose_graph_path, "w") as file:
        for i in range(len(coordinates)):
            tile_id = coordinates[i][0]
            filename = get_tile_filename(tile_id)
            # if not registration_results[filename].success:
            #     continue

            # write the node
            center = coordinates[i][1]
            file.write(
                f"VERTEX_SE3:QUAT_TIME {tile_id} {center[0]:.2f} {center[1]:.2f} {center[2]:.2f} 0 0 0 1 0 0\n"
            )
        # Edges
        saved_edges = []
        for i in range(len(coordinates)):
            tile_id = coordinates[i][0]
            filename = get_tile_filename(tile_id)
            # if not registration_results[filename].success:
            #     continue
            # write the edge
            # 4 neighbours
            col = i // grid_size_row
            row = i % grid_size_row

            neighbours_row = [-1, 0, 0, 1]
            neighbours_col = [0, -1, 1, 0]

            # write the prior edge, between tile and aerial cloud
            tile_center = coordinates[i][1]
            tile_pose = np.eye(4)
            tile_pose[0:3, 3] = tile_center
            mat = registration_results[filename].transform

            # transform tile center to uav in world frame
            transformed_tile_pose = mat @ tile_pose
            quat = rotation_matrix_to_quat(
                transformed_tile_pose[0:3, 0:3]
            )  # x, y, z, w
            file.write(
                f"EDGE_SE3:QUAT {tile_id} {tile_id} {transformed_tile_pose[0, 3]:.2f} {transformed_tile_pose[1, 3]:.2f} {transformed_tile_pose[2, 3]:.2f} {quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f} 1e+06 0 0 0 0 0 1e+06 0 0 0 0 1e+06 0 0 0 10000 0 0 10000 0 10000\n"
            )

            for j in range(4):
                neighbour_row = row + neighbours_row[j]
                neighbour_col = col + neighbours_col[j]
                if (
                    neighbour_row >= 0
                    and neighbour_row < grid_size_row
                    and neighbour_col >= 0
                    and neighbour_col < grid_size_col
                ):

                    neighbour = neighbour_col * grid_size_row + neighbour_row
                    neighbour_id = coordinates[neighbour][0]
                    # neighbour_filename = get_tile_filename(neighbour_id)
                    # if not registration_results[neighbour_filename].success:
                    #     continue

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
