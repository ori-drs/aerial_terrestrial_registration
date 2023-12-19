import numpy as np
from scipy.spatial.transform import Rotation


def euler_to_rotation_matrix(yaw, pitch, roll):
    # Create rotation matrices for each axis
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    # Combine the rotation matrices in ZYX order (yaw-pitch-roll)
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))

    return rotation_matrix


def rotation_matrix_to_quat(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a (x,y,z,w) quaternion"""
    r = Rotation.from_matrix(rotation_matrix)
    return r.as_quat()
