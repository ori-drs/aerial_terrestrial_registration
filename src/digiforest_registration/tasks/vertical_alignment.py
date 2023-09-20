import numpy as np
from digiforest_analysis.tasks import GroundSegmentation


class VerticalRegistration:
    def __init__(self, uav_cloud, bls_cloud, ground_segmentation_method):
        self._max_distance_to_plane = 0.5
        self.ground_segmentation = GroundSegmentation(
            max_distance_to_plane=self._max_distance_to_plane,
            cell_size=4.0,
            normal_thr=0.92,
            box_size=80,
            method=ground_segmentation_method,
        )
        self.uav_cloud = uav_cloud
        self.bls_cloud = bls_cloud
        self.debug = False

    def process(self):
        ground_uav_cloud, _ = self.ground_segmentation.process(cloud=self.uav_cloud)
        ground, _ = self.ground_segmentation.process(cloud=self.bls_cloud)

        # segment the two ground planes
        plane_model_uav, inliers_uav = ground_uav_cloud.to_legacy().segment_plane(
            distance_threshold=self._max_distance_to_plane,
            ransac_n=30,
            num_iterations=1000,
        )
        [a_r, b_r, c_r, d_r] = plane_model_uav
        n_r = np.array([a_r, b_r, c_r])
        norm_r = np.linalg.norm(n_r)
        n_r = n_r / np.linalg.norm(n_r)

        plane_model, inliers = ground.to_legacy().segment_plane(
            distance_threshold=self._max_distance_to_plane,
            ransac_n=30,
            num_iterations=1000,
        )
        [a, b, c, d] = plane_model
        n = np.array([a, b, c])
        norm = np.linalg.norm(n)
        n = n / np.linalg.norm(n)

        if self.debug:
            # visualize the two ground planes
            inlier_cloud = ground.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

            inlier_cloud_uav = ground_uav_cloud.select_by_index(inliers_uav)
            inlier_cloud_uav.paint_uniform_color([1.0, 0.0, 0])

            import open3d as o3d

            o3d.visualization.draw_geometries(
                [inlier_cloud.to_legacy(), inlier_cloud_uav.to_legacy()]
            )

        # todo find rotation between the two normal vectors

        print([a_r, b_r, c_r, d_r], [a, b, c, d])
        print("dot product of normals: ", np.dot(n_r, n))
        print("Distance between planes: ", np.abs(d_r / norm_r - d / norm))
        z_offset = -(d_r / norm_r - d / norm)
        print("z offset bls cloud to uav cloud: ", z_offset)
        return [a_r, b_r, c_r, d_r], [a, b, c, d], z_offset
