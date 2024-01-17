import gtsam
import numpy as np
import open3d as o3d
import copy


class PoseGraph:
    def __init__(self):
        self._nodes = {}
        self._initial_nodes = {}
        self._edges = []
        self._adjacency = {}
        self._clouds = {}
        self._downsampled_clouds = {}
        self._cloud_names = {}
        self.root_id = None

    @property
    def size(self):
        return len(self._nodes)

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @edges.setter
    def edges(self, edges):
        self._edges = edges

    def get_node_pose(self, id):
        return self._nodes[id]["pose"]

    def get_initial_node_pose(self, id):
        if id in self._initial_nodes:
            return self._initial_nodes[id]["pose"]
        else:
            return self._nodes[id]["pose"]

    def get_node_cloud_name(self, id):
        try:
            return self._cloud_names[id]
        except Exception:
            return str()

    # def _transform_node_cloud(self, cloud, node_id: int):
    #     """
    #     Transform the cloud to center it with the node"""
    #     center = get_cloud_center(cloud)
    #     center_pose = np.eye(4)
    #     center_pose[0:3, 3] = center
    #     node_pose = self.get_node_pose(node_id)
    #     # transform cloud to node pose
    #     cloud.transform(node_pose.matrix() @ np.linalg.inv(center_pose))
    #     return cloud

    def get_node_cloud_downsampled(self, id):
        """
        Return the downsampled cloud attached to the node
        """
        try:
            cloud = self._downsampled_clouds[id]
            return cloud
        except Exception:
            return o3d.geometry.PointCloud()

    def get_node_cloud(self, id):
        """
        Return the cloud attached to the node
        """
        try:
            cloud = self._clouds[id]
            return cloud
        except Exception:
            return o3d.geometry.PointCloud()

    def set_node_pose(self, id, pose):
        if id not in self._initial_nodes:
            # initializing initial node poses
            self._initial_nodes[id] = copy.deepcopy(self._nodes[id])
        self._nodes[id]["pose"] = pose

    # def _is_valid_id(self, id):
    #     return id >= 0 and id < self.size

    def add_node(self, id, stamp, pose):
        assert isinstance(pose, gtsam.Pose3)
        if self.root_id is None:
            self.root_id = id
        self._nodes[id] = {"pose": pose, "stamp": stamp, "id": id}
        self._adjacency[id] = {}

    def add_edge(self, parent_id, child_id, edge_type, relative_pose, relative_info):
        # This adds directed edges, even though they should be an undirected graph
        # We do this to simplify the API
        assert isinstance(relative_pose, gtsam.Pose3)
        assert isinstance(relative_info, np.ndarray)
        assert relative_info.shape == (6, 6)

        # if not self._is_valid_id(parent_id):
        #     raise KeyError(
        #         f"Node parent [{parent_id}] not in graph. Cannot add the edge"
        #     )
        # if not self._is_valid_id(child_id):
        #     raise KeyError(f"Node child [{child_id}] not in graph. Cannot add the edge")

        self._edges.append(
            {
                "parent_id": parent_id,
                "child_id": child_id,
                "pose": relative_pose,
                "info": relative_info,
                "type": edge_type,
            }
        )
        # Save reference to edge in adjacency matrix
        self._adjacency[parent_id][child_id] = len(self._edges) - 1

    def add_clouds(self, id: int, scan, scan_name: str):
        assert isinstance(scan, o3d.cuda.pybind.t.geometry.PointCloud)
        self._clouds[id] = scan
        self._downsampled_clouds[id] = scan.voxel_down_sample(voxel_size=0.2)
        self._cloud_names[id] = scan_name
