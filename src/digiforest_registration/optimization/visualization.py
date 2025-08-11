import copy
import numpy as np
import open3d as o3d

# colors
RED = [1, 0.0, 0.0]
GRAY = [0.3, 0.3, 0.3]
LIGHT_GRAY = [0.7, 0.7, 0.7]
ORANGE = [1.0, 0.5, 0.0]
GREEN = [0.0, 1, 0.0]
COLORS = [RED, ORANGE, LIGHT_GRAY, GREEN]


def graph_to_geometries(
    graph,
    show_frames=True,
    show_edges=True,
    show_nodes=True,
    show_clouds=False,
    show_coordinate_frame=True,
    odometry_color=GRAY,
    loop_color=RED,
):

    pose_graph = copy.deepcopy(graph)
    geometries = []
    node_centers = []
    frames_vis = o3d.geometry.TriangleMesh()
    nodes_vis = o3d.geometry.TriangleMesh()

    node_id_to_index = {}
    for id, node in pose_graph.nodes.items():

        pose = node["pose"].matrix()
        pos = pose[0:3, 3]
        rot = pose[0:3, 0:3]
        node_id_to_index[id] = len(node_centers)
        node_centers.append(pos)

        if show_frames:
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=2.0, origin=pos
            )
            frame_mesh.rotate(rot, center=pos)
            frames_vis += frame_mesh

        if show_nodes:
            node_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
            node_mesh.translate(pos)
            nodes_vis += node_mesh

        if show_clouds:
            try:
                cloud = pose_graph.get_node_cloud_downsampled(id)

                cloud.paint_uniform_color(COLORS[id % len(COLORS)])
                geometries.append(cloud.to_legacy())
            except Exception:
                pass

    if len(node_centers) == 0:
        return geometries

    if show_coordinate_frame:
        axes_center = node_centers[0] + np.array([2, 2, 0])  # arbitrary offset
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=3.0, origin=axes_center
        )
        geometries.append(axes)
    # vis.add_geometry("clouds", clouds_vis)
    geometries.append(frames_vis)
    geometries.append(nodes_vis)

    if show_edges:
        edges = []
        edge_colors = []
        for e in pose_graph.edges:
            if e["type"] != "in-between":
                continue

            edges.append(
                [node_id_to_index[e["parent_id"]], node_id_to_index[e["child_id"]]]
            )
            edge_colors.append(
                odometry_color if e["type"] == "in-between" else loop_color
            )

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(node_centers),
            lines=o3d.utility.Vector2iVector(edges),
        )
        line_set.colors = o3d.utility.Vector3dVector(edge_colors)
        geometries.append(line_set)

    return geometries
