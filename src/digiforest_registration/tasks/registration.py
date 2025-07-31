#!/usr/bin/env python3
from digiforest_registration.tasks.vertical_alignment import VerticalRegistration
from digiforest_registration.tasks.horizontal_alignment import HorizontalRegistration
from digiforest_registration.tasks.icp import icp
from digiforest_registration.utils import euler_to_rotation_matrix
from digiforest_registration.utils import crop_cloud
from digiforest_registration.utils import ExperimentLogger

import numpy as np
import open3d as o3d


class RegistrationResult:
    def __init__(self):
        self.success = False
        self.transform = None
        self.icp_fitness = 0.0


class Registration:
    def __init__(
        self,
        uav_cloud,
        mls_cloud,
        ground_segmentation_method,
        correspondence_matching_method,
        mls_feature_extraction_method,
        icp_fitness_threshold,
        min_distance_between_peaks,
        max_number_of_clique,
        logging_dir: str,
        debug=False,
    ):
        self.uav_cloud = uav_cloud
        self.mls_cloud = mls_cloud  # shouldn't modify the input cloud
        self.ground_segmentation_method = ground_segmentation_method
        self.correspondence_matching_method = correspondence_matching_method
        self.mls_feature_extraction_method = mls_feature_extraction_method
        self.debug = debug
        self.icp_fitness_threshold = icp_fitness_threshold
        self.min_distance_between_peaks = min_distance_between_peaks
        self.max_number_of_clique = max_number_of_clique
        self.transform = np.identity(4)
        self.success = False
        self.report = {"icp_fitness": 0, "clique_size": 0}  # TODO: replace with logger
        self.logger = ExperimentLogger(base_dir=logging_dir)

    def find_transform(self, horizontal_registration, transform: np.ndarray) -> float:
        best_icp_fitness_score = 0
        # Iterate through all the transformations ( one for each maximum clique )
        # and find the one that gives the best icp fitness score

        for i in range(len(horizontal_registration.transforms)):
            mls_cloud = self.mls_cloud.clone()
            M = horizontal_registration.transforms[i]
            tx = M[0, 2]
            ty = M[1, 2]
            yaw = np.arctan2(M[1, 0], M[0, 0])
            self.logger.debug(
                f"Transformation from mls cloud to uav (x, y, yaw, scale): {tx}, {ty}, {yaw}"
            )

            R = euler_to_rotation_matrix(yaw, 0, 0)
            transform[0:3, 0:3] = R
            transform[0, 3] = tx
            transform[1, 3] = ty

            mls_cloud.transform(transform)
            if self.debug:
                # Visualize the results
                mls_cloud.paint_uniform_color([0.8, 0.8, 0.8])
                self.uav_cloud.paint_uniform_color([0.0, 1.0, 0])
                o3d.visualization.draw_geometries(
                    [mls_cloud.to_legacy(), self.uav_cloud.to_legacy()],
                    window_name="Result after horizontal alignment",
                )

            # Crop the uav cloud around the mls cloud and reestimate the transformation
            # along the z axis
            cropped_uav_cloud = crop_cloud(self.uav_cloud, mls_cloud, padding=1)
            vertical_registration = VerticalRegistration(
                cropped_uav_cloud,
                mls_cloud,
                ground_segmentation_method=self.ground_segmentation_method,
                logger=self.logger,
                debug=self.debug,
            )
            (_, _, tz) = vertical_registration.process()
            vertical_transform = np.identity(4)
            vertical_transform[2, 3] = tz

            mls_cloud.transform(vertical_transform)
            transform[2, 3] = tz + transform[2, 3]
            self.logger.debug("Transformation matrix before icp:")
            self.logger.debug(transform)

            # Use cropped uav cloud in the rest of the code
            if self.debug:
                o3d.visualization.draw_geometries(
                    [mls_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
                    window_name="Result before ICP",
                )

            # Apply final icp registration
            icp_transform, icp_fitness = icp(mls_cloud, cropped_uav_cloud, self.logger)
            mls_cloud.transform(icp_transform)
            if self.debug:
                o3d.visualization.draw_geometries(
                    [mls_cloud.to_legacy(), cropped_uav_cloud.to_legacy()],
                    window_name="Final registration",
                )

            self.logger.debug("Final transformation matrix:")
            self.logger.debug(icp_transform @ transform)

            if icp_fitness >= best_icp_fitness_score:
                best_icp_fitness_score = icp_fitness
                self.transform = icp_transform @ transform

            if best_icp_fitness_score > 0.95:
                # we are happy with the result
                break

        self.report["icp_fitness"] = best_icp_fitness_score
        self.report["clique_size"] = horizontal_registration.clique_size
        return best_icp_fitness_score

    def registration(self) -> bool:
        # we are estimating the transformation matrix from mls cloud to uav cloud
        transform = np.identity(4)
        vertical_registration = VerticalRegistration(
            self.uav_cloud,
            self.mls_cloud,
            ground_segmentation_method=self.ground_segmentation_method,
            logger=self.logger,
            debug=self.debug,
        )
        try:
            (uav_ground_plane, mls_ground_plane, tz) = vertical_registration.process()
        except Exception as e:
            self.logger.error(f"Vertical registration failed: {e}")
            return False
        transform[2, 3] = tz

        # find the x, y, yaw transformation
        horizontal_registration = HorizontalRegistration(
            self.uav_cloud,
            uav_ground_plane,
            self.mls_cloud,
            mls_ground_plane,
            min_distance_between_peaks=self.min_distance_between_peaks,
            max_number_of_clique=self.max_number_of_clique,
            correspondence_matching_method=self.correspondence_matching_method,
            mls_feature_extraction_method=self.mls_feature_extraction_method,
            logger=self.logger,
            debug=self.debug,
        )
        self.success = horizontal_registration.process()
        if not self.success:
            return False

        icp_fitness = self.find_transform(horizontal_registration, transform)

        return icp_fitness > self.icp_fitness_threshold

    def colorize_cloud(self, cloud, icp_fitness):
        import matplotlib.pyplot as plt

        colormap = plt.get_cmap("RdYlBu")

        num_colors = 10
        values = np.linspace(0, 1, num_colors)
        # values = np.flip(values)  # to have the blue color for the best fitness

        index = np.floor(icp_fitness * num_colors).astype(int)
        cloud.paint_uniform_color(colormap(values[index])[:3])

    def transform_cloud(self, cloud):
        """
        Transform the cloud using the estimated transformation matrix"""

        transformed_cloud = cloud.clone()
        if not self.success:
            self.colorize_cloud(transformed_cloud, 0.0)
            return transformed_cloud

        transformed_cloud.transform(self.transform)
        self.colorize_cloud(transformed_cloud, self.report["icp_fitness"])
        return transformed_cloud
