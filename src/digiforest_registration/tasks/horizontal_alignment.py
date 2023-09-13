import numpy as np
from numpy.typing import NDArray
from numpy import float64
import cv2
import open3d as o3d
import copy
import networkx as nx

from digiforest_registration.tasks.height_image import HeightImage
from digiforest_registration.tasks.graph import Graph, CorrespondenceGraph


class HorizontalRegistration:
    def __init__(
        self, reference_cloud, reference_ground_plane, cloud, cloud_ground_plane
    ):
        self.reference_cloud = reference_cloud
        self.reference_ground_plane = reference_ground_plane
        self.cloud = cloud
        self.cloud_ground_plane = cloud_ground_plane

        self.min_distance_to_ground = 3.0
        self.image_resolution = 0.1

    def point_plane_distance(self, a, b, c, d, array: NDArray[float64]):
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
            v = max(0, dist[index][0] / max_height[0])
            image[y, x] = [max(v, image[y, x])]
            index += 1

        return image

    def find_feature_matches(self, img1, img2):
        # Initiate detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 10 matches.
        cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            matches[:10],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries(
            [source_temp, target_temp],
            zoom=0.4459,
            front=[0.9288, -0.2951, -0.2242],
            lookat=[1.6784, 2.0612, 1.4451],
            up=[-0.3402, -0.9189, -0.1996],
        )

    def registration(self, ref_pts, pts):
        """
        Run icp on the two input clouds, doesn't work so far
        """
        ref_cloud = o3d.geometry.PointCloud()
        ref_cloud.points = o3d.utility.Vector3dVector(ref_pts)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)

        threshold = 0.02
        trans_init = np.asarray(
            [[1.0, 0, 0, 245], [0, 1.0, 0, 187], [0, 0, 1.0, 0], [0.0, 0.0, 0.0, 1.0]]
        )
        self.draw_registration_result(cloud, ref_cloud, trans_init)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            cloud,
            ref_cloud,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        self.draw_registration_result(cloud, ref_cloud, reg_p2p.transformation)

    def process(self):
        uav_image = self.compute_canopy_image(
            self.reference_cloud, *self.reference_ground_plane
        )
        image = self.compute_canopy_image(self.cloud, *self.cloud_ground_plane)

        # find maxima in the heigh image
        proc = HeightImage()
        image_height_pts = proc.find_local_maxima(image)

        uav_height_pts = proc.find_local_maxima(uav_image)

        # create feature graphs
        G = Graph(image_height_pts, node_prefix="f")
        H = Graph(uav_height_pts, node_prefix="uav")

        correspondence_graph = CorrespondenceGraph(G, H)
        print("Computing the maximum clique")
        max_clique = nx.algorithms.approximation.max_clique(correspondence_graph.graph)

        # Print the maximum cliques
        print(max_clique)

        # mcs = nx.algorithms.isomorphism.ISMAGS(G.graph, H.graph, edge_match=compare_edge)
        # generator = mcs.largest_common_subgraph()
        # for v in generator:
        #     print(v)
