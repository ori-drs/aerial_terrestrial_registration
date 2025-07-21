import numpy as np
from digiforest_analysis.tasks import GroundSegmentation

import open3d as o3d


class VerticalRegistration:
    def __init__(self, uav_cloud, mls_cloud, ground_segmentation_method, debug=False):
        self._max_distance_to_plane = 0.5
        self.ground_segmentation = GroundSegmentation(
            max_distance_to_plane=self._max_distance_to_plane,
            cell_size=4.0,
            normal_thr=0.92,
            box_size=80,
            method=ground_segmentation_method,
        )
        self.uav_cloud = uav_cloud
        self.mls_cloud = mls_cloud
        self.debug = debug

    def project_point_onto_plane(self, point, plane_normal, plane_constant):
        """
        Project a 3D point onto a plane.

        Args:
            point (numpy.ndarray): The 3D point to be projected, e.g., [x, y, z].
            plane_normal (numpy.ndarray): The normal vector of the plane, e.g., [a, b, c].
            plane_constant (float): The constant term 'd' in the plane equation, e.g., d.

        Returns:
            numpy.ndarray: The projected point on the plane.
        """
        # Calculate the signed distance from the point to the plane
        distance = np.dot(point, plane_normal) + plane_constant

        # Calculate the projection of the point onto the plane
        projected_point = point - distance * plane_normal

        return projected_point

    def process(self):
        ground_uav_cloud, _ = self.ground_segmentation.process(cloud=self.uav_cloud)
        ground, _ = self.ground_segmentation.process(cloud=self.mls_cloud)

        # segment the two ground planes
        plane_model_uav, inliers_uav = ground_uav_cloud.to_legacy().segment_plane(
            distance_threshold=self._max_distance_to_plane,
            ransac_n=30,
            num_iterations=1000,
        )
        [a_r, b_r, c_r, d_r] = plane_model_uav
        n_r = np.array([a_r, b_r, c_r])
        n_r = n_r / np.linalg.norm(n_r)

        plane_model, inliers = ground.to_legacy().segment_plane(
            distance_threshold=self._max_distance_to_plane,
            ransac_n=30,
            num_iterations=1000,
        )
        [a, b, c, d] = plane_model
        n = np.array([a, b, c])
        n = n / np.linalg.norm(n)

        if self.debug:
            # visualize the two ground planes
            inlier_cloud = ground.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

            inlier_cloud_uav = ground_uav_cloud.select_by_index(inliers_uav)
            inlier_cloud_uav.paint_uniform_color([0, 0.0, 1.0])

            o3d.visualization.draw_geometries(
                [inlier_cloud.to_legacy(), inlier_cloud_uav.to_legacy()],
                window_name="ground point clouds",
            )

        # todo find rotation between the two normal vectors

        print([a_r, b_r, c_r, d_r], [a, b, c, d])
        print("dot product of normals: ", np.dot(n_r, n))

        uav_point = (
            ground_uav_cloud.select_by_index(inliers_uav)
            .point.positions.numpy()
            .mean(axis=0)
        )
        p_proj = self.project_point_onto_plane(uav_point, n, d)
        z_offset = np.sign(uav_point[2] - p_proj[2]) * np.linalg.norm(
            p_proj - uav_point
        )
        print("Signed Distance between planes", z_offset)

        return [a_r, b_r, c_r, d_r], [a, b, c, d], z_offset
