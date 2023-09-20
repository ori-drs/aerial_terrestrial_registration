from digiforest_registration.tasks.height_image import HeightImage, draw_correspondences
from digiforest_registration.tasks.graph import Graph, CorrespondenceGraph

import numpy as np
import cv2


class HorizontalRegistration:
    def __init__(self, uav_cloud, uav_ground_plane, cloud, cloud_ground_plane):
        self.uav_cloud = uav_cloud
        self.uav_ground_plane = uav_ground_plane
        self.cloud = cloud
        self.cloud_ground_plane = cloud_ground_plane
        self.debug = True

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

        if self.debug:
            draw_correspondences(
                bls_height_img, bls_height_pts, uav_height_img, uav_height_pts, edges
            )

        # find transformation using maximum clique
        bls_pts = np.zeros((len(edges), 2))
        uav_pts = np.zeros((len(edges), 2))
        for i in range(len(edges)):
            bls_pts[i] = bls_proc.pixel_to_utm(edges[i][0][0], edges[i][0][1])
            uav_pts[i] = uav_proc.pixel_to_utm(edges[i][1][0], edges[i][1][1])

        M = cv2.estimateAffine2D(bls_pts, uav_pts)[0]
        tx = M[0, 2]
        ty = M[1, 2]
        yaw = np.arctan2(M[1, 0], M[0, 0])
        scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)

        print(
            "Transformation from bls cloud to uav (x, y, yaw, scale):",
            tx,
            ty,
            yaw,
            scale,
        )

        return tx, ty, yaw
