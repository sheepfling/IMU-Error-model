import numpy as np


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    angle = float(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    if angle < 1e-8:
        return 0.5 * skew
    if np.pi - angle < 1e-6:
        diagonal = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        index = int(np.argmax(axis))
        if axis[index] < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            if index == 0:
                axis[1] = rotation[0, 1] / (2.0 * axis[0])
                axis[2] = rotation[0, 2] / (2.0 * axis[0])
            elif index == 1:
                axis[0] = rotation[0, 1] / (2.0 * axis[1])
                axis[2] = rotation[1, 2] / (2.0 * axis[1])
            else:
                axis[0] = rotation[0, 2] / (2.0 * axis[2])
                axis[1] = rotation[1, 2] / (2.0 * axis[2])
            axis /= np.linalg.norm(axis)
        return angle * axis
    return angle / (2.0 * np.sin(angle)) * skew


def validate_orientation(orientation: np.ndarray) -> np.ndarray:
    orientation = np.asarray(orientation, dtype=float)
    if orientation.shape != (3, 3):
        raise ValueError("orientation must have shape (3, 3)")
    if not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(orientation), 1.0, atol=1e-7):
        raise ValueError("orientation must be a proper rotation matrix")
    return orientation
