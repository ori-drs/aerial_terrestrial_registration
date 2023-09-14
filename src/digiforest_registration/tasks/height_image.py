import numpy as np
import cv2


class HeightImage:
    def __init__(self):
        self.kernel_size = (3, 3)

    def remove_small_objects(self, img):
        # remove small non-black objects from a float image
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
        return filtered_img

    def find_local_maxima(self, img):
        def distance_to_border(img, y, x):
            """Distance from to the nearest border"""
            return min(y, img.shape[0] - y, x, img.shape[1] - x)

        # the input image is a float image
        # find local maxima in the image

        # preprocessing the image by remove noise
        img = self.remove_small_objects(img)

        kernel_size = (25, 25)
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

        # cv2.imshow("Image", maxima_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return np.array(valid_points), img


def draw_correspondences(height_img1, height_pts1, height_img2, height_pts2, edges):
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
    # Draw the edges
    for edge in edges:
        cv2.line(
            img, (edge[0][0], edge[0][1]), (edge[1][0] + w1, edge[1][1]), (0, 255, 0), 1
        )

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
