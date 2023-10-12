import open3d as o3d
import numpy as np


def icp(source, target):
    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    max_correspondence_distance = 0.5

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    # Convergence-Criteria for Vanilla ICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=50
    )

    init_source_to_target = np.identity(4)
    registration_icp = o3d.pipelines.registration.registration_icp(
        source.to_legacy(),
        target.to_legacy(),
        max_correspondence_distance,
        init_source_to_target,
        estimation,
        criteria,
    )

    print("Inlier Fitness: ", registration_icp.fitness)
    print("Inlier RMSE: ", registration_icp.inlier_rmse)
    print("ICP transform is:")
    print(registration_icp.transformation)
    return registration_icp.transformation
