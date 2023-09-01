import numpy as np
import cv2


class HeightImage:
    def __init__(self):
        self.kernel_size = (3, 3)

    def remove_small_objects(self, img):
        # remove small objects from a float image
        grayscale_image = (img * 255).astype(np.uint8)

        threshold = 1
        _, binary_mask = cv2.threshold(
            grayscale_image, threshold, 255, cv2.THRESH_BINARY
        )

        kernel = np.ones(self.kernel_size, np.uint8)
        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        opening = opening.astype(np.float32)

        filtered_img = np.multiply(img, opening)
        filtered_img = filtered_img[:, :, 0]
        cv2.imshow("img", img)
        cv2.imshow("opening", opening)
        cv2.imshow("filtered_img", filtered_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return opening

    def find_local_maxima(self, img):
        def distance_to_border(img, y, x):
            """Distance from to the nearest border"""
            return min(y, img.shape[0] - y, x, img.shape[1] - x)

        # the input image is a float image
        # find local maxima in the image

        # preprocessing the image by remove noise
        # img = self.remove_small_objects(img)

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

        valid_points = []
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
        return np.array(valid_points)
