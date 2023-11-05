from digiforest_registration.tasks.height_image import HeightImage, draw_correspondences
from digiforest_registration.tasks.graph import Graph, CorrespondenceGraph

import numpy as np

from typing import Tuple


class HorizontalRegistration:
    def __init__(
        self, uav_cloud, uav_ground_plane, cloud, cloud_ground_plane, debug=False
    ):
        self.uav_cloud = uav_cloud
        self.uav_ground_plane = uav_ground_plane
        self.cloud = cloud
        self.cloud_ground_plane = cloud_ground_plane
        self.debug = debug

    def find_transform(self, src, dst, estimate_scale=False):
        """Estimate N-D similarity transformation with or without scaling.

        Parameters
        ----------
        src : (M, N) array_like
            Source coordinates.
        dst : (M, N) array_like
            Destination coordinates.
        estimate_scale : bool
            Whether to estimate scaling factor.

        Returns
        -------
        T : (N + 1, N + 1)
            The homogeneous similarity transformation matrix. The matrix contains
            NaN values only if the problem is not well-conditioned.

        Source:
        https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py

        """
        src = np.asarray(src)
        dst = np.asarray(dst)

        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        # Eq. (38).
        A = dst_demean.T @ src_demean / num

        # Eq. (39).
        d = np.ones((dim,), dtype=np.float64)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.float64)

        U, S, V = np.linalg.svd(A)

        # Eq. (40) and (43).
        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.nan * T
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ np.diag(d) @ V

        if estimate_scale:
            # Eq. (41) and (42).
            scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
        else:
            scale = 1.0

        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
        T[:dim, :dim] *= scale

        return T

    def process(self) -> Tuple[bool, float, float, float]:
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
        print("Creating the feature graphs")
        G = Graph(bls_height_pts, node_prefix="f")
        H = Graph(uav_height_pts, node_prefix="uav")

        print("Number of nodes of the frontier graph", G.graph.number_of_nodes())
        print("Number of nodes of the uav graph", H.graph.number_of_nodes())

        # np.savetxt('/tmp/frontier_peaks.txt', bls_height_pts, delimiter=",", fmt='%.4f')

        # find maximum clique in the correspondence graph
        correspondence_graph = CorrespondenceGraph(G, H)
        print("Computing the maximum clique")
        edges = correspondence_graph.maximum_clique()

        if self.debug:
            draw_correspondences(
                bls_height_img, bls_height_pts, uav_height_img, uav_height_pts, edges
            )

        # find transformation using maximum clique
        bls_pts = np.zeros((len(edges), 2))
        uav_pts = np.zeros((len(edges), 2))
        for i in range(len(edges)):
            bls_pts[i] = bls_proc.pixel_to_cloud(edges[i][0][0], edges[i][0][1])
            uav_pts[i] = uav_proc.pixel_to_cloud(edges[i][1][0], edges[i][1][1])

        if bls_pts.shape[0] < 3:
            return False, 0, 0, 0

        M = self.find_transform(bls_pts, uav_pts)
        tx = M[0, 2]
        ty = M[1, 2]
        yaw = np.arctan2(M[1, 0], M[0, 0])

        print("Transformation from bls cloud to uav (x, y, yaw, scale):", tx, ty, yaw)

        return True, tx, ty, yaw
