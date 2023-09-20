from digiforest_registration.tasks.height_image import HeightImage, draw_correspondences
from digiforest_registration.tasks.graph import Graph, CorrespondenceGraph

import numpy as np


class HorizontalRegistration:
    def __init__(self, uav_cloud, uav_ground_plane, cloud, cloud_ground_plane):
        self.uav_cloud = uav_cloud
        self.uav_ground_plane = uav_ground_plane
        self.cloud = cloud
        self.cloud_ground_plane = cloud_ground_plane

    def estimate_similarity_transformation_with_known_correspondences(self, p1, p2):

        tx, ty, theta, s = None, None, None, None
        if p1.shape[0] < 2 or p1.shape != p2.shape:
            print("Need at least 2 points to compute similarity transformation!")
            return tx, ty, theta, s

        # Get the number of matches
        k = p1.shape[1]

        # Compute mu_x1, mu_y1, mu_x2, mu_y2
        mu_x1 = np.sum(p1[0, :])
        mu_y1 = np.sum(p1[1, :])
        mu_x2 = np.sum(p2[0, :])
        mu_y2 = np.sum(p2[1, :])

        # Compute l_1p2, l_1m2
        l_1p2 = np.sum(p1[0, :] * p2[0, :] + p1[1, :] * p2[1, :])
        l_1m2 = np.sum(p1[0, :] * p2[1, :] - p1[1, :] * p2[0, :])

        # Compute l_11
        l_11 = np.sum(p1[0, :] ** 2 + p1[1, :] ** 2)

        # Compute det_r
        det_r = k * l_11 - mu_x1**2 - mu_y1**2

        # Compute M_P1, M_P2
        M_P1 = np.array(
            [
                [l_11, 0, -mu_x1, mu_y1],
                [0, l_11, -mu_y1, -mu_x1],
                [-mu_x1, -mu_y1, k, 0],
                [mu_y1, -mu_x1, 0, k],
            ]
        )
        M_P2 = np.array([mu_x2, mu_y2, l_1p2, l_1m2])

        # Compute registration parameters
        r = 1 / det_r * np.dot(M_P1, M_P2)

        # Set output parameters
        tx = r[0]
        ty = r[1]
        theta = np.arctan2(r[3], r[2])
        s = r[2] / np.cos(theta)

        return tx, ty, theta, s

    def process(self):
        uav_proc = HeightImage()
        bls_proc = HeightImage()

        uav_canopy = uav_proc.compute_canopy_image(
            self.uav_cloud, *self.uav_ground_plane
        )
        bls_canopy = bls_proc.compute_canopy_image(self.cloud, *self.cloud_ground_plane)

        # find maxima in the heigh image
        bls_height_pts, bls_height_img = uav_proc.find_local_maxima(bls_canopy)

        uav_height_pts, uav_height_img = bls_proc.find_local_maxima(uav_canopy)

        # create feature graphs
        G = Graph(bls_height_pts, node_prefix="f")
        H = Graph(uav_height_pts, node_prefix="uav")

        # find maximum clique in the correspondence graph
        correspondence_graph = CorrespondenceGraph(G, H)
        print("Computing the maximum clique")
        edges = correspondence_graph.maximum_clique()
        print(edges)

        draw_correspondences(
            bls_height_img, bls_height_pts, uav_height_img, uav_height_pts, edges
        )

        # find transformation using maximum clique
        bls_pts = np.zeros((len(edges), 2))
        uav_pts = np.zeros((len(edges), 2))
        for i in range(len(edges)):
            bls_pts[i] = bls_proc.pixel_to_utm(edges[i][0][0], edges[i][0][1])
            uav_pts[i] = uav_proc.pixel_to_utm(edges[i][1][0], edges[i][1][1])

        (
            tx,
            ty,
            theta,
            s,
        ) = self.estimate_similarity_transformation_with_known_correspondences(
            bls_pts, uav_pts
        )

        print("Transformation from bls cloud to uav:", tx, ty, theta)
