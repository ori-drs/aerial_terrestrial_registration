#!/usr/bin/env python3
from digiforest_registration.tasks.vertical_alignment import VerticalRegistration
from digiforest_registration.tasks.horizontal_alignment import HorizontalRegistration
from digiforest_registration.tasks.icp import icp
from digiforest_registration.utils import euler_to_rotation_matrix

import numpy as np
import open3d as o3d


class Registration:
    def __init__(self, uav_cloud, frontier_cloud, ground_segmentation_method):
        self.uav_cloud = uav_cloud
        self.frontier_cloud = frontier_cloud
        self.ground_segmentation_method = ground_segmentation_method

    def crop_cloud(self, uav_cloud, frontier_cloud, padding):
        """
        Crop the uav cloud around the frontier cloud and return the cropped cloud
        """
        bbox = frontier_cloud.get_axis_aligned_bounding_box()
        frontier_min_bound = bbox.min_bound.numpy()
        frontier_max_bound = bbox.max_bound.numpy()

        large_z_padding = 10**10
        min_bound = o3d.core.Tensor(
            [
                frontier_min_bound[0] - padding,
                frontier_min_bound[1] - padding,
                -large_z_padding,
            ],
            dtype=o3d.core.Dtype.Float32,
        )
        max_bound = o3d.core.Tensor(
            [
                frontier_max_bound[0] + padding,
                frontier_max_bound[1] + padding,
                large_z_padding,
            ],
            dtype=o3d.core.Dtype.Float32,
        )
        crop_box = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        cropped_uav_cloud = uav_cloud.crop(crop_box)
        return cropped_uav_cloud

    def registration(self) -> bool:
        # transformation matrix from frontier cloud to uav cloud that we are estimating
        transform = np.identity(4)
        vertical_registration = VerticalRegistration(
            self.uav_cloud,
            self.frontier_cloud,
            ground_segmentation_method=self.ground_segmentation_method,
        )
        (uav_groud_plane, frontier_ground_plane, tz) = vertical_registration.process()
        transform[2, 3] = tz

        ##############################

        horizontal_registration = HorizontalRegistration(
            self.uav_cloud, uav_groud_plane, self.frontier_cloud, frontier_ground_plane
        )
        (tx, ty, yaw) = horizontal_registration.process()

        R = euler_to_rotation_matrix(yaw, 0, 0)
        transform[0:3, 0:3] = R
        transform[0, 3] = tx
        transform[1, 3] = ty

        # Visualize the results
        self.frontier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        self.uav_cloud.paint_uniform_color([0.0, 1.0, 0])

        self.frontier_cloud.transform(transform)
        # o3d.visualization.draw_geometries(
        #     [frontier_cloud.to_legacy(), uav_cloud.to_legacy()]
        # )

        # Crop the uav cloud around the frontier cloud and reestimate the transformation
        # along the z axis
        cropped_uav_cloud = self.crop_cloud(
            self.uav_cloud, self.frontier_cloud, padding=4
        )
        vertical_registration = VerticalRegistration(
            cropped_uav_cloud,
            self.frontier_cloud,
            ground_segmentation_method=self.ground_segmentation_method,
        )
        (_, _, tz) = vertical_registration.process()
        vertical_transform = np.identity(4)
        vertical_transform[2, 3] = tz
        print("Old z offset", transform[2, 3], ", new z offset", tz + transform[2, 3])
        self.frontier_cloud.transform(vertical_transform)
        transform[2, 3] = tz + transform[2, 3]
        print("Transformation matrix before icp:")
        print(transform)

        # Use cropped uav cloud in the rest of the code

        o3d.visualization.draw_geometries(
            [self.frontier_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
            window_name="Result before ICP",
        )

        # Apply final icp registration
        icp_transform, icp_fitness = icp(self.frontier_cloud, cropped_uav_cloud)
        self.frontier_cloud.transform(icp_transform)
        o3d.visualization.draw_geometries(
            [self.frontier_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
            window_name="Final registration",
        )

        print("Final transformation matrix:")
        np.set_printoptions(suppress=True)
        print(icp_transform @ transform)
        if icp_fitness < 0.5:
            return False
        return True
