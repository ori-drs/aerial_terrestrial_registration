import numpy as np
from digiforest_analysis.tasks.tree_segmentation import TreeSegmentation


class TreeTrunkSegmentation:
    def __init__(self, debug: bool = False):
        self.max_distance_to_plane = 0.5
        self.debug = debug
        # TODO set debug_level according to debug
        self.tree_seg = TreeSegmentation(debug_level=0, clustering_method="voronoi")

    def _point_plane_distance(
        self, a: float, b: float, c: float, d: float, array
    ) -> float:
        """
        Calculate the distance between a point and a plane
        ax + by + cz + d = 0
        """
        normal = np.tile(np.array([a, b, c]), (array.shape[0], 1))
        d_array = np.tile(d, (array.shape[0], 1))

        # Calculate the distance using the formula
        dot_product = np.sum(array * normal, axis=1)
        dot_product = dot_product[:, np.newaxis]  # make it a column vector
        numerator = np.abs(dot_product + d_array)
        denominator = np.linalg.norm(np.array([a, b, c]))

        distance = numerator / denominator
        return distance

    def find_tree_trunks(self, cloud, groud_plane) -> np.ndarray:
        """
        ground_plane is the equation of the ground plane in the form [a, b, c, d]
        Returns an array of tree trunk positions"""
        # assert isinstance(cloud, o3d.cuda.pybind.t.geometry.PointCloud)

        points = np.asarray(cloud.to_legacy().points)
        dist = self._point_plane_distance(
            groud_plane[0], groud_plane[1], groud_plane[2], groud_plane[3], points
        )
        idx = (dist > 0) & (dist < 0.2)
        idx = idx.flatten()  # make it a row vector
        ground_points = points[idx]
        # z_ground = ground_points[0][
        #     2
        # ]
        # TODO can improve how the z coordinate of the ground is detected
        z_ground = np.mean(ground_points[:, 2])

        cloud_translated = cloud.clone()
        cloud_translated.translate([0, 0, -z_ground])

        # segment stems
        clusters = self.tree_seg.process(
            cloud=cloud_translated,
            cloth=None,
            max_cluster_radius=2,
            n_threads=8,
            point_fraction=0.1,
            # crop_lower_bound=z_ground + 4,
            # crop_upper_bound=z_ground + 6,
            crop_lower_bound=4,
            crop_upper_bound=6,
        )

        trunks_positions = []
        for cluster in clusters:
            transform = cluster["info"]["axis"]["transform"]
            trunks_positions.append([transform[0, 3], transform[1, 3], 0])

        return np.array(trunks_positions)
