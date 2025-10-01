import gtsam
import numpy as np
import copy


class PoseGraph:
    """A class to represent a pose graph"""

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

    def get_node_id_from_stamp(self, stamp):
        for node_id in self._nodes:
            if self._nodes[node_id]["stamp"] == stamp:
                return node_id

        raise ValueError("Unknown stamp")

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

    def get_node_cloud_downsampled(self, id):
        """
        Return the downsampled cloud attached to the node
        """
        cloud = self._downsampled_clouds[id]
        return cloud

    def get_node_cloud(self, id):
        """
        Return the cloud attached to the node
        """
        cloud = self._clouds[id]
        return cloud

    def set_node_pose(self, id, pose):
        if id not in self._initial_nodes:
            # initializing initial node poses
            self._initial_nodes[id] = copy.deepcopy(self._nodes[id])
        self._nodes[id]["pose"] = pose

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
        # assert isinstance(scan, o3d.cuda.pybind.t.geometry.PointCloud)
        self._clouds[id] = scan
        self._downsampled_clouds[id] = scan.voxel_down_sample(voxel_size=0.2)
        self._cloud_names[id] = scan_name
