from numpy import allclose, arccos, argmax, array, asarray, clip, diag, eye, isclose, maximum, ndarray, pi, sin, sqrt, \
    trace as matrix_trace
from numpy.linalg import det, norm


def rotation_vector_from_matrix(rotation: ndarray) -> ndarray:
    """Return the axis-angle rotation vector represented by a 3×3 matrix."""
    trace = float(matrix_trace(rotation))
    angle = float(arccos(clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    skew = array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    if angle < 1e-8:
        return 0.5 * skew
    ####
    if pi - angle < 1e-6:
        diagonal = maximum((diag(rotation) + 1.0) / 2.0, 0.0)
        axis = sqrt(diagonal)
        index = int(argmax(axis))
        if axis[index] < 1e-8:
            axis = array([1.0, 0.0, 0.0])
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
            ####
            axis /= norm(axis)
        ####
        return angle * axis
    ####
    return angle / (2.0 * sin(angle)) * skew
####

def validate_orientation(orientation: ndarray) -> ndarray:
    """Validate and return a proper 3×3 world-from-body rotation matrix."""
    orientation = asarray(orientation, dtype=float)
    if orientation.shape != (3, 3):
        raise ValueError("orientation must have shape (3, 3)")
    ####
    if (not allclose(orientation.T @ orientation, eye(3), atol=1e-7) or
            not isclose(det(orientation), 1.0, atol=1e-7)):
        raise ValueError("orientation must be a proper rotation matrix")
    ####
    return orientation
####
