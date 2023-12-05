import open3d as o3d
import numpy as np


def crop_cloud(cloud_to_be_cropped, cloud, padding):
    """
    Crop the cloud_to_be_cropped around the cloud and return the cropped cloud
    """
    bbox = cloud.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound.numpy()
    max_bound = bbox.max_bound.numpy()

    large_z_padding = 10**10
    min_bound = o3d.core.Tensor(
        [
            min_bound[0] - padding,
            min_bound[1] - padding,
            -large_z_padding,
        ],
        dtype=o3d.core.Dtype.Float32,
    )
    max_bound = o3d.core.Tensor(
        [
            max_bound[0] + padding,
            max_bound[1] + padding,
            large_z_padding,
        ],
        dtype=o3d.core.Dtype.Float32,
    )
    crop_box = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    cropped_cloud = cloud_to_be_cropped.crop(crop_box)
    return cropped_cloud


def crop_cloud_to_size(cloud, size):
    """
    Crop the cloud to make it square
    """
    bbox = cloud.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound.numpy()
    max_bound = bbox.max_bound.numpy()

    padding_x = max((max_bound[0] - min_bound[0] - size) / 2, 0)
    padding_y = max((max_bound[1] - min_bound[1] - size) / 2, 0)
    z_padding = 10**10
    min_bound = o3d.core.Tensor(
        [
            min_bound[0] + padding_x,
            min_bound[1] + padding_y,
            -z_padding,
        ],
        dtype=o3d.core.Dtype.Float32,
    )
    max_bound = o3d.core.Tensor(
        [
            max_bound[0] - padding_x,
            max_bound[1] - padding_y,
            z_padding,
        ],
        dtype=o3d.core.Dtype.Float32,
    )
    crop_box = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    cropped_cloud = cloud.crop(crop_box)
    return cropped_cloud


def get_cloud_center(cloud) -> np.ndarray:
    """
    Returns the center of a point cloud."""
    # get bounding box
    bbox = cloud.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound.numpy()
    max_bound = bbox.max_bound.numpy()
    # get center of cloud
    center = (min_bound + max_bound) / 2
    return center
