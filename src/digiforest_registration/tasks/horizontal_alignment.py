from digiforest_registration.tasks.height_image import HeightImage, draw_correspondences
from digiforest_registration.tasks.graph import Graph, CorrespondenceGraph


class HorizontalRegistration:
    def __init__(
        self, reference_cloud, reference_ground_plane, cloud, cloud_ground_plane
    ):
        self.reference_cloud = reference_cloud
        self.reference_ground_plane = reference_ground_plane
        self.cloud = cloud
        self.cloud_ground_plane = cloud_ground_plane
        self.debug = False

    def process(self):
        proc = HeightImage()
        uav_canopy = proc.compute_canopy_image(
            self.reference_cloud, *self.reference_ground_plane
        )
        bls_canopy = proc.compute_canopy_image(self.cloud, *self.cloud_ground_plane)

        # find maxima in the heigh image
        bls_height_pts, bls_height_img = proc.find_local_maxima(bls_canopy)

        uav_height_pts, uav_height_img = proc.find_local_maxima(uav_canopy)

        # create feature graphs
        G = Graph(bls_height_pts, node_prefix="f")
        H = Graph(uav_height_pts, node_prefix="uav")

        correspondence_graph = CorrespondenceGraph(G, H)
        print("Computing the maximum clique")
        edges = correspondence_graph.maximum_clique()
        print(edges)

        draw_correspondences(
            bls_height_img, bls_height_pts, uav_height_img, uav_height_pts, edges
        )
