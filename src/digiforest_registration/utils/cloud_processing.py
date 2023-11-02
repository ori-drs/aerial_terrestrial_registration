import open3d as o3d


def crop_cloud(cloud_to_be_cropped, cloud, padding):
    """
    Crop the cloud_to_be_cropped around the cloud and return the cropped cloud
    """
    bbox = cloud.get_axis_aligned_bounding_box()
    frontier_min_bound = bbox.min_bound.numpy()
    frontier_max_bound = bbox.max_bound.numpy()

    large_z_padding = 10**10
    min_bound = o3d.core.Tensor(
        [
            frontier_min_bound[0] - padding,
            frontier_min_bound[1] - padding,
            -large_z_padding,
        ],
        dtype=o3d.core.Dtype.Float32,
    )
    max_bound = o3d.core.Tensor(
        [
            frontier_max_bound[0] + padding,
            frontier_max_bound[1] + padding,
            large_z_padding,
        ],
        dtype=o3d.core.Dtype.Float32,
    )
    crop_box = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    cropped_cloud = cloud_to_be_cropped.crop(crop_box)
    return cropped_cloud
