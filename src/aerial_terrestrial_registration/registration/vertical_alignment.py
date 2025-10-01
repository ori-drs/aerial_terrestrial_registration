import numpy as np
from digiforest_analysis.tasks import GroundSegmentation
from aerial_terrestrial_registration.utils import CloudIO
import open3d as o3d


class VerticalRegistration:
    """Class to perform vertical registration of UAV and MLS point clouds."""

    def __init__(
        self,
        uav_cloud,
        mls_cloud,
        cloud_io: CloudIO,
        ground_segmentation_method,
        logger,
        debug=False,
    ):
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
        self.cloud_io = cloud_io
        self.logger = logger  # TODO is it possible to make a no-op logger ?
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
        """Process the UAV and MLS point clouds to find the vertical transformation.

        Returns:
            A tuple containing the equations of the ground plane of the two clouds
            and the offset between these two planes
            tuple:
                plane (list[float]): [a, b, c, d] coefficients of the aerial ground plane equation
                plane (list[float]): [a, b, c, d] coefficients of the MLS ground plane equation
                z offset (float): signed distance between the two planes
        """
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

        # visualize the two ground planes
        inlier_cloud = ground.select_by_index(inliers)

        inlier_cloud_uav = ground_uav_cloud.select_by_index(inliers_uav)

        if self.debug:
            o3d.visualization.draw_geometries(
                [inlier_cloud.to_legacy(), inlier_cloud_uav.to_legacy()],
                window_name="ground point clouds",
            )

        # TODO if the two normal vectors are not collinear
        # find rotation matrix between the two

        uav_point = (
            ground_uav_cloud.select_by_index(inliers_uav)
            .point.positions.numpy()
            .mean(axis=0)
        )
        p_proj = self.project_point_onto_plane(uav_point, n, d)
        z_offset = np.sign(uav_point[2] - p_proj[2]) * np.linalg.norm(
            p_proj - uav_point
        )

        if self.logger:
            self.logger.log_pointcloud(inlier_cloud, self.cloud_io, "mls_ground_cloud")
            self.logger.log_pointcloud(
                inlier_cloud_uav, self.cloud_io, "uav_ground_cloud"
            )
            self.logger.debug(f"f{[a_r, b_r, c_r, d_r]}, {[a, b, c, d]}")
            self.logger.debug(f"dot product of normals: {np.dot(n_r, n)}")
            self.logger.debug(f"Signed Distance between planes {z_offset}")

        return [a_r, b_r, c_r, d_r], [a, b, c, d], z_offset
