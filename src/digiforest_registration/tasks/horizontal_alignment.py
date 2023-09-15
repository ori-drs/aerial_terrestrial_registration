import numpy as np
from numpy.typing import NDArray
from numpy import float64

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

        self.min_distance_to_ground = 3.0
        self.image_resolution = 0.1

    def point_plane_distance(self, a, b, c, d, array: NDArray[float64]) -> float:
        """
        Calculate the distance between a point and a plane
        ax + by + cz + d = 0
        """
        normal = np.tile(np.array([a, b, c]), (array.shape[0], 1))
        d_array = np.tile(d, (array.shape[0], 1))

        # Calculate the distance using the formula
        dot_product = np.sum(array * normal, axis=1)
        dot_product = dot_product[:, np.newaxis]  # make it a column vector
        numerator = np.abs(dot_product + d_array)
        denominator = np.linalg.norm(np.array([a, b, c]))

        distance = numerator / denominator
        return distance

    def compute_canopy_image(self, cloud, a, b, c, d):
        points = np.asarray(cloud.to_legacy().points)
        dist = self.point_plane_distance(a, b, c, d, points)
        idx = dist > self.min_distance_to_ground
        idx = idx.flatten()  # make it a row vector
        canopy_points = points[idx]

        if self.debug:
            # Display canopy points as a point cloud
            import open3d as o3d

            print(a, b, c, d)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(canopy_points)
            o3d.visualization.draw_geometries([point_cloud])

        bounding_box = cloud.get_axis_aligned_bounding_box()

        # Get the minimum and maximum points of the bounding box
        min_bound = bounding_box.min_bound.numpy()
        max_bound = bounding_box.max_bound.numpy()

        # Create the canopy image
        height = int(np.ceil((max_bound[1] - min_bound[1]) / self.image_resolution))
        width = int(np.ceil((max_bound[0] - min_bound[0]) / self.image_resolution))
        print("height, width", height, width)

        image = np.zeros((height, width, 1), dtype=np.float32)

        # Fill the image
        dist = self.point_plane_distance(a, b, c, d, canopy_points)
        max_height = max(dist)
        index = 0
        for point in canopy_points:
            x = int(np.floor((point[0] - min_bound[0]) / self.image_resolution))
            y = int(np.floor((point[1] - min_bound[1]) / self.image_resolution))
            v = abs(dist[index][0] / max_height[0])
            image[y, x] = max(v, image[y, x])
            index += 1

        return image

    def process(self):
        uav_canopy = self.compute_canopy_image(
            self.reference_cloud, *self.reference_ground_plane
        )
        bls_canopy = self.compute_canopy_image(self.cloud, *self.cloud_ground_plane)

        # find maxima in the heigh image
        proc = HeightImage()
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
