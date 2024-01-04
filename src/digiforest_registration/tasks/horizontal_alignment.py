from digiforest_registration.tasks.height_image import HeightImage, draw_correspondences
from digiforest_registration.tasks.graph import Graph, CorrespondenceGraph

import numpy as np


class HorizontalRegistration:
    def __init__(
        self, uav_cloud, uav_ground_plane, cloud, cloud_ground_plane, debug=False
    ):
        self.uav_cloud = uav_cloud
        self.uav_ground_plane = uav_ground_plane
        self.cloud = cloud
        self.cloud_ground_plane = cloud_ground_plane
        self.debug = debug
        self.transforms = []
        self.max_number_of_clique = 5
        self.clique_size = 0
        self.frontier_peaks_size = 0

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

    def process(self) -> bool:
        uav_proc = HeightImage(self.debug)
        bls_proc = HeightImage(self.debug)

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
        # edge1 = G.graph.get_edge_data('f_1', 'f_7')
        H = Graph(uav_height_pts, node_prefix="uav")

        print("Number of nodes of the frontier graph", G.graph.number_of_nodes())
        print("Number of nodes of the uav graph", H.graph.number_of_nodes())

        # np.savetxt('/tmp/frontier_peaks.txt', bls_height_pts, delimiter=",", fmt='%.4f')

        # find maximum clique in the correspondence graph
        correspondence_graph = CorrespondenceGraph(G, H)

        print("Computing the maximum clique")
        correspondences_list = correspondence_graph.maximum_clique()

        if len(correspondences_list) > self.max_number_of_clique:
            # too many cliques, something is wrong
            print("Too many cliques, downsampling them")
            # TODO it's not great
            correspondences_list = correspondences_list[0 : self.max_number_of_clique]
            # return False
        elif len(correspondences_list) == 0:
            return False

        for i in range(len(correspondences_list)):
            correspondences = correspondences_list[i]
            if self.debug:
                draw_correspondences(
                    bls_height_img,
                    bls_height_pts,
                    uav_height_img,
                    uav_height_pts,
                    correspondences,
                )

            # find transformation using maximum clique
            bls_pts = np.zeros((len(correspondences), 2))
            uav_pts = np.zeros((len(correspondences), 2))
            for i in range(len(correspondences)):
                bls_pts[i] = bls_proc.pixel_to_cloud(
                    correspondences[i][0][0], correspondences[i][0][1]
                )
                uav_pts[i] = uav_proc.pixel_to_cloud(
                    correspondences[i][1][0], correspondences[i][1][1]
                )

            if bls_pts.shape[0] < 3:
                return False

            M = self.find_transform(bls_pts, uav_pts)
            self.transforms.append(M)

        return True
