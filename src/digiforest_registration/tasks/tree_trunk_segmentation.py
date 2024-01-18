import numpy as np
import open3d as o3d

from digiforest_analysis.tasks.tree_segmentation import TreeSegmentation


class TreeTrunkSegmentation:
    def __init__(self, debug: bool = False):
        self.max_distance_to_plane = 0.5
        self.debug = debug
        # TODO set debug_level according to debug
        self.tree_seg = TreeSegmentation(debug_level=0, clustering_method="voronoi")

    def find_tree_trunks(self, cloud, groud_plane) -> np.ndarray:
        """
        ground_plane is the equation of the ground plane in the form [a, b, c, d]
        Returns an array of tree trunk positions"""
        assert isinstance(cloud, o3d.cuda.pybind.t.geometry.PointCloud)

        [_, _, _, d] = groud_plane

        # segment stems
        clusters = self.tree_seg.process(
            cloud=cloud,
            cloth=None,
            max_cluster_radius=2,
            n_threads=8,
            point_fraction=0.1,
            crop_lower_bound=-d + 4,
            crop_upper_bound=-d + 6,
        )

        trunks_positions = []
        for cluster in clusters:
            transform = cluster["info"]["axis"]["transform"]
            trunks_positions.append([transform[0, 3], transform[1, 3], 0])

        return np.array(trunks_positions)
