from digiforest_registration.tasks.height_image import (
    HeightImage,
    CanopyFeatureDescriptor,
    CanopyFeatureMatcher,
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
        cloud,
        cloud_ground_plane,
        debug=False,
        correspondence_matching_method="graph",
        bls_feature_extraction_method="canopy_map",
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
        self.feature_association_method = (
            correspondence_matching_method  # graph or feature_extraction
        )
        self.bls_feature_extraction_method = (
            bls_feature_extraction_method  # canopy_map or tree_segmentation
        )

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
        bls_height_pts, bls_height_img = bls_proc.find_local_maxima(bls_canopy)

        uav_height_pts, uav_height_img = uav_proc.find_local_maxima(uav_canopy)

        if self.bls_feature_extraction_method == "tree_segmentation":
            # find tree trunks
            tree_trunk_segmentation = TreeTrunkSegmentation()
            bls_pts = tree_trunk_segmentation.find_tree_trunks(
                self.cloud, self.cloud_ground_plane
            )
            bounding_box = self.cloud.get_axis_aligned_bounding_box()
            # convert to pixel coordinates
            bls_height_pts2 = np.zeros((bls_pts.shape[0], 2), dtype=np.int32)
            for i in range(bls_pts.shape[0]):
                bls_height_pts2[i] = bls_proc.cloud_point_to_pixel(
                    bls_pts[i], bounding_box, image_resolution=0.1
                )

            bls_height_pts = bls_height_pts2  # TODO improve that
        elif self.bls_feature_extraction_method != "canopy_map":
            raise ValueError("Unknown method: " + self.bls_feature_extraction_method)

        # import pickle
        # pickle.dump(bls_height_pts, open("/tmp/bls_height_pts.pkl", "wb"))
        # pickle.dump(bls_height_img, open("/tmp/bls_height_img.pkl", "wb"))
        # pickle.dump(uav_height_pts, open("/tmp/uav_height_pts.pkl", "wb"))
        # pickle.dump(uav_height_img, open("/tmp/uav_height_img.pkl", "wb"))

        if self.feature_association_method == "graph":

            # create feature graphs
            print("Creating the feature graphs")
            G = Graph(bls_height_pts, node_prefix="f")
            H = Graph(uav_height_pts, node_prefix="uav")

            print(
                "Number of nodes and edges of the frontier graph",
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
            if self.bls_feature_extraction_method == "tree_segmentation":
                correspondence_graph.distance_threshold = 0.25

            print("Computing the maximum clique")
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

        elif self.feature_association_method == "feature_extraction":
            # find correspondences using feature extraction
            print("Computing correspondences using feature extraction")
            descriptor = CanopyFeatureDescriptor()
            uav_descriptors = descriptor.compute_feature_descriptors(uav_height_pts)
            bls_descriptors = descriptor.compute_feature_descriptors(bls_height_pts)
            feature_matcher = CanopyFeatureMatcher()
            correspondences_list = feature_matcher.match(
                bls_height_pts, bls_descriptors, uav_height_pts, uav_descriptors
            )
        else:
            raise ValueError("Unknown method: " + self.feature_association_method)

        for i in range(len(correspondences_list)):
            correspondences = correspondences_list[i]
            if self.debug:
                draw_correspondences(
                    bls_height_img,
                    bls_height_pts,
                    uav_height_img,
                    uav_height_pts,
                    correspondences,
                    True,
                    G,
                    H,
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

            # need at least 3 pairs of points to find a transformation
            if bls_pts.shape[0] < 3:
                return False

            M = self.find_transform(bls_pts, uav_pts)
            self.transforms.append(M)

        return True

    def _test_edges(self, G, H, correspondence_graph):
        edges = []
        # edges.append([('f_13', 'f_15'), ('uav_5', 'uav_6' )])
        # edges.append([('f_15', 'f_19'), ('uav_6', 'uav_7' )])
        # edges.append([('f_10', 'f_13'), ('uav_8', 'uav_5' )])
        # edges.append([('f_10', 'f_3'), ('uav_8', 'uav_10' )])
        # edges.append([('f_3', 'f_17'), ('uav_10', 'uav_11' )])
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


if __name__ == "__main__":
    import pickle

    bls_height_pts = pickle.load(open("/tmp/bls_height_pts.pkl", "rb"))
    uav_height_pts = pickle.load(open("/tmp/uav_height_pts.pkl", "rb"))
    bls_height_img = pickle.load(open("/tmp/bls_height_img.pkl", "rb"))
    uav_height_img = pickle.load(open("/tmp/uav_height_img.pkl", "rb"))

    descriptor = CanopyFeatureDescriptor()
    uav_descriptors = descriptor.compute_feature_descriptors(uav_height_pts)
    bls_descriptors = descriptor.compute_feature_descriptors(bls_height_pts)
    feature_matcher = CanopyFeatureMatcher()
    correspondences_list = feature_matcher.match(
        bls_height_pts, bls_descriptors, uav_height_pts, uav_descriptors
    )

    for i in range(len(correspondences_list)):
        correspondences = correspondences_list[i]

        draw_correspondences(
            bls_height_img,
            bls_height_pts,
            uav_height_img,
            uav_height_pts,
            correspondences,
        )
