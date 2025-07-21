from digiforest_registration.tasks.height_image import (
    HeightImage,
    display_correspondences,
    draw_correspondences,
)
from digiforest_registration.tasks.graph import Graph, CorrespondenceGraph
from digiforest_registration.tasks.tree_trunk_segmentation import TreeTrunkSegmentation

import numpy as np


class HorizontalRegistration:
    def __init__(
        self,
        uav_cloud,
        uav_ground_plane,
        mls_cloud,
        mls_cloud_ground_plane,
        min_distance_between_peaks,
        max_number_of_clique,
        logger,
        debug=False,
        correspondence_matching_method="graph",
        mls_feature_extraction_method="canopy_map",
    ):
        self.uav_cloud = uav_cloud
        self.uav_ground_plane = uav_ground_plane
        self.mls_cloud = mls_cloud
        self.mls_cloud_ground_plane = mls_cloud_ground_plane
        self.debug = debug
        self.logger = logger
        self.transforms = []
        self.max_number_of_clique = max_number_of_clique
        self.clique_size = 0
        self.feature_association_method = (
            correspondence_matching_method  # graph or feature_extraction
        )
        self.mls_feature_extraction_method = (
            mls_feature_extraction_method  # canopy_map or tree_segmentation
        )
        self.min_distance_between_peaks = min_distance_between_peaks

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

        uav_proc = HeightImage(
            min_distance_between_peaks=self.min_distance_between_peaks,
            logger=self.logger,
            debug=self.debug,
        )
        mls_proc = HeightImage(
            min_distance_between_peaks=self.min_distance_between_peaks,
            logger=self.logger,
            debug=self.debug,
        )

        uav_canopy = uav_proc.compute_canopy_image(
            self.uav_cloud, *self.uav_ground_plane
        )
        mls_canopy = mls_proc.compute_canopy_image(
            self.mls_cloud, *self.mls_cloud_ground_plane
        )

        # find maxima in the heigh image
        mls_height_pts, mls_height_img = mls_proc.find_local_maxima(mls_canopy)

        uav_height_pts, uav_height_img = uav_proc.find_local_maxima(uav_canopy)

        if self.mls_feature_extraction_method == "tree_segmentation":
            # find tree trunks
            tree_trunk_segmentation = TreeTrunkSegmentation()
            mls_pts = tree_trunk_segmentation.find_tree_trunks(
                self.cloud, self.cloud_ground_plane
            )
            bounding_box = self.cloud.get_axis_aligned_bounding_box()
            # convert to pixel coordinates
            mls_height_pts = np.zeros((mls_pts.shape[0], 2), dtype=np.int32)
            for i in range(mls_pts.shape[0]):
                mls_height_pts[i] = mls_proc.cloud_point_to_pixel(
                    mls_pts[i], bounding_box, image_resolution=0.1
                )

        elif self.mls_feature_extraction_method != "canopy_map":
            raise ValueError("Unknown method: " + self.mls_feature_extraction_method)

        # save points to disk
        # self._save_tree_location_to_disk(uav_proc, uav_height_pts, mls_proc, mls_height_pts)

        # import pickle
        # pickle.dump(mls_height_pts, open("/tmp/mls_height_pts.pkl", "wb"))
        # pickle.dump(mls_height_img, open("/tmp/mls_height_img.pkl", "wb"))
        # pickle.dump(uav_height_pts, open("/tmp/uav_height_pts.pkl", "wb"))
        # pickle.dump(uav_height_img, open("/tmp/uav_height_img.pkl", "wb"))

        if self.feature_association_method == "graph":

            # create feature graphs
            print("Creating the feature graphs")
            G = Graph(mls_height_pts, node_prefix="f")
            H = Graph(uav_height_pts, node_prefix="uav")

            print(
                "Number of nodes and edges of the mls graph",
                G.graph.number_of_nodes(),
                G.graph.number_of_edges(),
            )
            print(
                "Number of nodes and edges of the uav graph",
                H.graph.number_of_nodes(),
                H.graph.number_of_edges(),
            )

            # find maximum clique in the correspondence graph
            correspondence_graph = CorrespondenceGraph(G, H)
            if self.mls_feature_extraction_method == "tree_segmentation":
                correspondence_graph.distance_threshold = 0.25

            print("Computing the maximum clique")
            if correspondence_graph.graph.number_of_edges() > 1800000:
                print("Too many edges in the correspondence graph")
                return False
            correspondences_list = correspondence_graph.maximum_clique()

            if len(correspondences_list) > self.max_number_of_clique:
                # too many cliques, something is wrong
                print("Too many cliques, downsampling them")
                # TODO it's not great
                correspondences_list = correspondences_list[
                    0 : self.max_number_of_clique
                ]
                # return False
            elif len(correspondences_list) == 0:
                return False

        else:
            raise ValueError("Unknown method: " + self.feature_association_method)

        for i in range(len(correspondences_list)):
            correspondences = correspondences_list[i]
            if self.debug:
                display_correspondences(
                    mls_height_img,
                    mls_height_pts,
                    uav_height_img,
                    uav_height_pts,
                    correspondences,
                    False,
                    G,
                    H,
                )
            correspondences_img = draw_correspondences(
                mls_height_img,
                mls_height_pts,
                uav_height_img,
                uav_height_pts,
                correspondences,
                False,
                G,
                H,
            )
            self.logger.log_image(correspondences_img, "correspondences")

            # find transformation using maximum clique
            mls_pts = np.zeros((len(correspondences), 2))
            uav_pts = np.zeros((len(correspondences), 2))
            for i in range(len(correspondences)):
                mls_pts[i] = mls_proc.pixel_to_cloud(
                    correspondences[i][0][0], correspondences[i][0][1]
                )
                uav_pts[i] = uav_proc.pixel_to_cloud(
                    correspondences[i][1][0], correspondences[i][1][1]
                )

            # need at least 3 pairs of points to find a transformation
            if mls_pts.shape[0] < 3:
                return False

            M = self.find_transform(mls_pts, uav_pts)
            self.transforms.append(M)

        return True

    # TODO delete
    def _test_edges(self, G, H, correspondence_graph):
        edges = []
        edges.append([("f_1", "f_2"), ("uav_31", "uav_36")])
        edges.append([("f_2", "f_0"), ("uav_36", "uav_26")])
        edges.append([("f_2", "f_4"), ("uav_36", "uav_42")])
        edges.append([("f_4", "f_6"), ("uav_42", "uav_51")])
        edges.append([("f_7", "f_10"), ("uav_63", "uav_69")])
        for edge in edges:
            edge1 = G.graph.get_edge_data(edge[0][0], edge[0][1])
            edge2 = H.graph.get_edge_data(edge[1][0], edge[1][1])
            print(
                G.pos[edge[0][0]],
                G.pos[edge[0][1]],
                G.get_angle(G.pos[edge[0][0]], G.pos[edge[0][1]]),
            )
            print(
                H.pos[edge[1][0]],
                H.pos[edge[1][1]],
                H.get_angle(H.pos[edge[1][0]], H.pos[edge[1][1]]),
            )
            print(
                correspondence_graph.compare_edge(
                    edge1, edge2, use_angle=True, debug=True
                )
            )

    # TODO delete
    def _save_tree_location_to_disk(
        self, uav_proc, uav_img_pts: np.ndarray, mls_proc, mls_img_pts: np.ndarray
    ):
        mls_pts = np.zeros((mls_img_pts.shape[0], 2))
        uav_pts = np.zeros((uav_img_pts.shape[0], 2))
        for i in range(mls_img_pts.shape[0]):
            mls_pts[i] = mls_proc.pixel_to_cloud(mls_img_pts[i][0], mls_img_pts[i][1])
        for i in range(uav_img_pts.shape[0]):
            uav_pts[i] = uav_proc.pixel_to_cloud(uav_img_pts[i][0], uav_img_pts[i][1])

        np.savetxt("/tmp/uav_pts.txt", uav_pts, fmt="%.2f", delimiter=",")
        np.savetxt("/tmp/mls_pts.txt", mls_pts, fmt="%.2f", delimiter=",")
