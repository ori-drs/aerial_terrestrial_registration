#!/usr/bin/env python3
from digiforest_registration.tasks.vertical_alignment import VerticalRegistration
from digiforest_registration.tasks.horizontal_alignment import HorizontalRegistration
from digiforest_registration.tasks.icp import icp
from digiforest_registration.utils import euler_to_rotation_matrix
from digiforest_registration.utils import crop_cloud

import numpy as np
import open3d as o3d


class Registration:
    def __init__(
        self, uav_cloud, frontier_cloud, ground_segmentation_method, debug=False
    ):
        self.uav_cloud = uav_cloud
        self.frontier_cloud = frontier_cloud
        self.ground_segmentation_method = ground_segmentation_method
        self.debug = debug
        self.icp_fitness_threshold = 0.85
        self.transform = None
        self.report = {}

    def registration(self) -> bool:
        # we are estimating the transformation matrix from frontier cloud to uav cloud
        transform = np.identity(4)
        vertical_registration = VerticalRegistration(
            self.uav_cloud,
            self.frontier_cloud,
            ground_segmentation_method=self.ground_segmentation_method,
            debug=self.debug,
        )
        (uav_groud_plane, frontier_ground_plane, tz) = vertical_registration.process()
        transform[2, 3] = tz

        ##############################

        horizontal_registration = HorizontalRegistration(
            self.uav_cloud,
            uav_groud_plane,
            self.frontier_cloud,
            frontier_ground_plane,
            debug=self.debug,
        )
        success = horizontal_registration.process()
        if not success:
            return False

        best_icp_fitness_score = 0
        for i in range(len(horizontal_registration.transforms)):
            frontier_cloud = self.frontier_cloud.clone()
            M = horizontal_registration.transforms[i]
            tx = M[0, 2]
            ty = M[1, 2]
            yaw = np.arctan2(M[1, 0], M[0, 0])
            print(
                "Transformation from bls cloud to uav (x, y, yaw, scale):", tx, ty, yaw
            )

            R = euler_to_rotation_matrix(yaw, 0, 0)
            transform[0:3, 0:3] = R
            transform[0, 3] = tx
            transform[1, 3] = ty

            frontier_cloud.transform(transform)
            if self.debug:
                # Visualize the results
                frontier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
                self.uav_cloud.paint_uniform_color([0.0, 1.0, 0])
                o3d.visualization.draw_geometries(
                    [frontier_cloud.to_legacy(), self.uav_cloud.to_legacy()],
                    window_name="Result after horizontal alignment",
                )

            # Crop the uav cloud around the frontier cloud and reestimate the transformation
            # along the z axis
            cropped_uav_cloud = crop_cloud(self.uav_cloud, frontier_cloud, padding=4)
            # vertical_registration = VerticalRegistration(
            #     cropped_uav_cloud,
            #     frontier_cloud,
            #     ground_segmentation_method=self.ground_segmentation_method,
            #     debug=self.debug,
            # )
            # (_, _, tz) = vertical_registration.process()
            # vertical_transform = np.identity(4)
            # vertical_transform[2, 3] = tz

            # frontier_cloud.transform(vertical_transform)
            # transform[2, 3] = tz + transform[2, 3]
            print("Transformation matrix before icp:")
            print(transform)

            # Use cropped uav cloud in the rest of the code
            if self.debug:
                o3d.visualization.draw_geometries(
                    [frontier_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
                    window_name="Result before ICP",
                )

            # Apply final icp registration
            icp_transform, icp_fitness = icp(frontier_cloud, cropped_uav_cloud)
            frontier_cloud.transform(icp_transform)
            if self.debug:
                o3d.visualization.draw_geometries(
                    [frontier_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
                    window_name="Final registration",
                )

            print("Final transformation matrix:")
            print(icp_transform @ transform)

            if icp_fitness > best_icp_fitness_score:
                best_icp_fitness_score = icp_fitness
                self.transform = icp_transform @ transform

        self.report["icp_fitness"] = icp_fitness
        self.report["clique_size"] = horizontal_registration.clique_size
        self.frontier_cloud.transform(self.transform)
        self.colorize_cloud(self.frontier_cloud, icp_fitness)

        return icp_fitness > self.icp_fitness_threshold

    def colorize_cloud(self, cloud, icp_fitness):
        import matplotlib.pyplot as plt

        colormap = plt.get_cmap("coolwarm")

        num_colors = 10
        values = np.linspace(0, 1, num_colors)
        values = np.flip(values)  # to have the blue color for the best fitness

        index = np.floor(icp_fitness * num_colors).astype(int)
        cloud.paint_uniform_color(colormap(values[index])[:3])
