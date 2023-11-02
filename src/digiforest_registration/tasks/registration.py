#!/usr/bin/env python3
from digiforest_registration.tasks.vertical_alignment import VerticalRegistration
from digiforest_registration.tasks.horizontal_alignment import HorizontalRegistration
from digiforest_registration.tasks.icp import icp
from digiforest_registration.utils import euler_to_rotation_matrix
from digiforest_registration.utils import crop_cloud

import numpy as np
import open3d as o3d


class Registration:
    def __init__(self, uav_cloud, frontier_cloud, ground_segmentation_method):
        self.uav_cloud = uav_cloud
        self.frontier_cloud = frontier_cloud
        self.ground_segmentation_method = ground_segmentation_method
        self.debug = False

    def registration(self):
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
        success, tx, ty, yaw = horizontal_registration.process()
        if not success:
            return None, False

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
        cropped_uav_cloud = crop_cloud(self.uav_cloud, self.frontier_cloud, padding=4)
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
        if self.debug:
            o3d.visualization.draw_geometries(
                [self.frontier_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
                window_name="Result before ICP",
            )

        # Apply final icp registration
        icp_transform, icp_fitness = icp(self.frontier_cloud, cropped_uav_cloud)
        self.frontier_cloud.transform(icp_transform)
        if self.debug:
            o3d.visualization.draw_geometries(
                [self.frontier_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
                window_name="Final registration",
            )

        print("Final transformation matrix:")
        print(icp_transform @ transform)

        return icp_transform @ transform, icp_fitness > 0.5
