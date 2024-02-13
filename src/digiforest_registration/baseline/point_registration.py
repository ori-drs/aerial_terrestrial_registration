import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import minimize

def load_coordinates(file_path):
    try:
        # Load data from text file
        data = np.loadtxt(file_path, delimiter=',')
        
        # Check if data has two columns (x, y coordinates)
        if data.shape[1] != 2:
            raise ValueError("The file should contain two columns for x and y coordinates.")
        
        return data
    except IOError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except ValueError as ve:
        print(f"Error: {ve}")
        return None

def find_correspondences(source_points, target_points):
    """
    Find the nearest neighbors between two point sets using KDTree.

    Args:
        source_points (numpy.ndarray): Array of source points.
        target_points (numpy.ndarray): Array of target points.

    Returns:
        list: List of tuples containing corresponding indices in source and target point sets.
    """
    tree = KDTree(target_points)
    distances, indices = tree.query(source_points)
    correspondences = [(i, indices[i]) for i in range(len(source_points))]
    return correspondences

def estimate_transform(source_points, target_points):
    """
    Estimate the transformation matrix between source and target point sets.

    Args:
        source_points (numpy.ndarray): Array of source points.
        target_points (numpy.ndarray): Array of target points.

    Returns:
        numpy.ndarray: 3x3 transformation matrix.
    """
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    H = np.dot(centered_source.T, centered_target)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_target - np.dot(R, centroid_source)

    transform = np.identity(3)
    transform[:2, :2] = R
    transform[:2, 2] = t
    return transform

def apply_transform(points, transform):
    """
    Apply the transformation matrix to a set of points.

    Args:
        points (numpy.ndarray): Array of points.
        transform (numpy.ndarray): 3x3 transformation matrix.

    Returns:
        numpy.ndarray: Transformed points.
    """
    homogenous_points = np.hstack([points, np.ones((len(points), 1))])
    transformed_points = np.dot(transform, homogenous_points.T).T
    return transformed_points[:, :2]

def icp(source_points, target_points, max_iterations=50, tolerance=1e-5):
    """
    Perform ICP (Iterative Closest Point) algorithm.

    Args:
        source_points (numpy.ndarray): Array of source points.
        target_points (numpy.ndarray): Array of target points.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence criteria.

    Returns:
        numpy.ndarray: Transformed source points.
        numpy.ndarray: Final transformation matrix.
    """
    transform = np.identity(3)
    for i in range(max_iterations):
        correspondences = find_correspondences(source_points, target_points)
        correspondences_source = np.array([source_points[idx] for idx, _ in correspondences])
        correspondences_target = np.array([target_points[idx] for _, idx in correspondences])
        
        new_transform = estimate_transform(correspondences_source, correspondences_target)
        source_points = apply_transform(source_points, new_transform)
        transform = np.dot(new_transform, transform)

        if np.abs(np.linalg.norm(new_transform) - 1) < tolerance:
            break

    return source_points, transform

# Load UAV points
uav_file_path = "data/test_evo/uav_pts.txt" 
uav_coordinates = load_coordinates(uav_file_path)
if uav_coordinates is not None:
    print("Loaded UAV coordinates:")
    print(uav_coordinates)


# Load BLS points
bls_file_path = "data/test_evo/bls_pts.txt" 
bls_coordinates = load_coordinates(bls_file_path)
if bls_coordinates is not None:
    print("Loaded BLS coordinates:")
    print(bls_coordinates)

# Compute transformation using icp
transformed_bls, final_transform = icp(bls_coordinates, uav_coordinates)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(uav_coordinates[:, 0], uav_coordinates[:, 1], color='blue', label='Source Points')
plt.scatter(bls_coordinates[:, 0], bls_coordinates[:, 1], color='red', marker='o', label='Target Points')
plt.scatter(transformed_bls[:, 0], transformed_bls[:, 1], color='green', marker='x', label='Transformed Target Points')
plt.title('ICP Transformation Result')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

print('Registration successful')