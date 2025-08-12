import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from numpy import float64

import cv2


class HeightImage:
    def __init__(self, min_distance_between_peaks: float, logger, debug: bool = False):
        self.kernel_size = (3, 3)
        self.debug = debug
        self.logger = logger
        self.min_distance_to_ground = 3.0
        self.image_resolution = 0.1  # meters per pixel
        # coordinates of the top left corner of the image in the utm frame
        self.top_left_corner = [0, 0, 0]
        self.min_distance_between_peaks = min_distance_between_peaks  # in meters

    @staticmethod
    def cloud_point_to_pixel(point, cloud_bounding_box, image_resolution: float):
        """
        Convert a point to a pixel coordinate"""
        min_bound = cloud_bounding_box.min_bound.numpy()
        max_bound = cloud_bounding_box.max_bound.numpy()
        height = int(np.ceil((max_bound[1] - min_bound[1]) / image_resolution))
        width = int(np.ceil((max_bound[0] - min_bound[0]) / image_resolution))
        x = max(
            0,
            min(
                int(np.floor((point[0] - min_bound[0]) / image_resolution)),
                width - 1,
            ),
        )
        y = max(
            0,
            min(
                int(np.floor((point[1] - min_bound[1]) / image_resolution)),
                height - 1,
            ),
        )
        return np.array([x, y], dtype=np.int32)

    def pixel_to_cloud(self, x, y):
        """
        Convert pixel coordinates to cloud coordinates"""

        x_cloud = x * self.image_resolution + self.top_left_corner[0]
        y_cloud = y * self.image_resolution + self.top_left_corner[1]
        return [x_cloud, y_cloud]

    @staticmethod
    def _point_plane_distance(
        a: float, b: float, c: float, d: float, array: NDArray[float64]
    ) -> float:
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

    def compute_canopy_image(
        self, cloud, a: float, b: float, c: float, d: float
    ) -> np.ndarray:
        """
        Compute the canopy image from a point cloud and a plane
        A canopy image is a 2D float image where the value of each pixel is the height of the canopy at this pixel
        """
        points = np.asarray(cloud.to_legacy().points)
        dist = self._point_plane_distance(a, b, c, d, points)
        idx = dist > self.min_distance_to_ground
        idx = idx.flatten()  # make it a row vector
        canopy_points = points[idx]

        bounding_box = cloud.get_axis_aligned_bounding_box()

        # Get the minimum and maximum points of the bounding box
        min_bound = bounding_box.min_bound.numpy()
        max_bound = bounding_box.max_bound.numpy()
        self.top_left_corner = min_bound

        # Create the canopy image
        height = int(np.ceil((max_bound[1] - min_bound[1]) / self.image_resolution))
        width = int(np.ceil((max_bound[0] - min_bound[0]) / self.image_resolution))
        self.logger.debug(f"height: {height}, width: {width}")

        image = np.zeros((height, width, 1), dtype=np.float32)

        # Fill the image
        dist = self._point_plane_distance(a, b, c, d, canopy_points)
        if len(dist) == 0:
            self.logger.warning("No points above the ground plane")
            return image

        max_height = max(dist)
        index = 0
        for point in canopy_points:
            x = min(
                int(np.floor((point[0] - min_bound[0]) / self.image_resolution)),
                width - 1,
            )
            y = min(
                int(np.floor((point[1] - min_bound[1]) / self.image_resolution)),
                height - 1,
            )
            v = abs(dist[index][0] / max_height[0])
            image[y, x] = max(v, image[y, x])
            index += 1

        return image

    def _remove_small_objects(self, img: np.ndarray) -> np.ndarray:
        """
        remove small non-black objects from a float image"""

        grayscale_image = (img * 255).astype(np.uint8)

        threshold = 1
        _, binary_mask = cv2.threshold(
            grayscale_image, threshold, 255, cv2.THRESH_BINARY
        )

        kernel = np.ones(self.kernel_size, np.uint8)
        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        opening = opening.astype(np.float32)
        opening = opening / 255.0

        # remove extra dimension
        img = np.squeeze(img)
        filtered_img = np.multiply(img, opening)

        self.logger.log_image(grayscale_image, "grayscale_canopy_image")
        if self.debug:
            cv2.imshow("img", img)
            cv2.imshow("grayscale_image", grayscale_image)
            cv2.imshow("filtered_img", filtered_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return filtered_img

    def find_local_maxima(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks in a float image.
        Return the coordinates of the peaks and the filtered image
        used to find the peaks"""

        def distance_to_border(img, y, x):
            """Distance from to the nearest border"""
            return min(y, img.shape[0] - y, x, img.shape[1] - x)

        # preprocessing the image by remove noise
        img = self._remove_small_objects(img)

        kernel_size = (
            int(self.min_distance_between_peaks / self.image_resolution),
            int(self.min_distance_between_peaks / self.image_resolution),
        )
        self.logger.debug(
            f"kernel_size {int(self.min_distance_between_peaks / self.image_resolution)}, min_distance_between_peaks {self.min_distance_between_peaks}"
        )
        kernel = np.ones(kernel_size, np.uint8)
        dilated_img = cv2.dilate(img, kernel)
        mask = cv2.compare(
            img, dilated_img, cv2.CMP_GE
        )  # set mask to 255 where img >= mask

        white_pixel_indices = np.where(mask == 255)
        white_points_coordinates = list(
            zip(white_pixel_indices[1], white_pixel_indices[0])
        )  # x, y coordinates

        #
        grayscale_image = (img * 255).astype(np.uint8)
        maxima_img = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

        valid_points = []  # (x, y)
        threshold_distance_to_border = 20
        for point in white_points_coordinates:
            if (
                grayscale_image[point[1], point[0]] > 0
                and distance_to_border(grayscale_image, point[1], point[0])
                > threshold_distance_to_border
            ):
                cv2.circle(maxima_img, point, 3, (0, 0, 255), -1)
                valid_points.append([point[0], point[1], 0])

        if self.debug:
            cv2.imshow("Image", maxima_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return np.array(valid_points), img


def draw_correspondences(
    height_img1: np.ndarray,
    height_pts1: np.ndarray,
    height_img2: np.ndarray,
    height_pts2: np.ndarray,
    correspondences: list,
    draw_node_names=False,
    graph1=None,
    graph2=None,
):
    grayscale_image = (height_img1 * 255).astype(np.uint8)
    img1 = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

    grayscale_image = (height_img2 * 255).astype(np.uint8)
    img2 = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

    # Combining the two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img = np.zeros((max(h1, h2), w1 + w2, 3), img1.dtype)
    img[:h1, :w1, :3] = img1
    img[:h2, w1 : (w1 + w2), :3] = img2

    # Draw the height points
    for point in height_pts1:
        cv2.circle(img, (point[0], point[1]), 3, (0, 0, 255), -1)
    for point in height_pts2:
        cv2.circle(img, (point[0] + w1, point[1]), 3, (0, 0, 255), -1)

    if draw_node_names:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        fontColor = (0, 255, 255)
        thickness = 1
        lineType = 2
        for i in range(len(height_pts1)):
            cv2.putText(
                img,
                graph1.node_name(i),
                (height_pts1[i][0], height_pts1[i][1]),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )

        for i in range(len(height_pts2)):
            cv2.putText(
                img,
                graph2.node_name(i),
                (height_pts2[i][0] + w1, height_pts2[i][1]),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )

    # Draw the correspondences
    for correspondence in correspondences:
        cv2.line(
            img,
            (correspondence[0][0], correspondence[0][1]),
            (correspondence[1][0] + w1, correspondence[1][1]),
            (0, 255, 0),
            1,
        )

    return img


def display_correspondences(
    height_img1: np.ndarray,
    height_pts1: np.ndarray,
    height_img2: np.ndarray,
    height_pts2: np.ndarray,
    correspondences: list,
    draw_node_names: bool = False,
    graph1=None,
    graph2=None,
):
    img = draw_correspondences(
        height_img1,
        height_pts1,
        height_img2,
        height_pts2,
        correspondences,
        draw_node_names,
        graph1,
        graph2,
    )

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
