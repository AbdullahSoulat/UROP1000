import numpy as np
# A tolerance for floating point comparisons
TOLERANCE = 1e-8

def vec_to_quat(v):
    """Converts a 3D vector to a pure unit quaternion."""
    q = np.zeros(4)
    q[1:] = v
    return q

def quat_to_vec(q):
    """Extracts the vector part of a quaternion."""
    return q[1:]

def quat_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_inverse(q):
    """Calculates the inverse of a quaternion."""
    # For a unit quaternion, the inverse is its conjugate
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_log(q):
    """Calculates the natural logarithm of a unit quaternion."""
    # Ensure the quaternion is a unit quaternion
    if abs(np.linalg.norm(q) - 1.0) > TOLERANCE:
        # Normalize to handle minor floating point inaccuracies
        q = q / np.linalg.norm(q)

    # If the quaternion is real (a scalar), the log is straightforward
    if np.linalg.norm(q[1:]) < TOLERANCE:
        # If q is (1,0,0,0), log is (0,0,0,0)
        if abs(q[0] - 1.0) < TOLERANCE:
            return np.array([0.0, 0.0, 0.0, 0.0])
        # If q is (-1,0,0,0), log is (0, pi, 0, 0) - or any direction for v
        # This case is tricky for standard log definition, usually avoided for rotations
        # For SQUAD, we typically deal with shortest paths, so q[0] should be positive.
        # If q[0] is -1, acos(-1) = pi. v/|v| can be anything. Let's pick (1,0,0) for example.
        if abs(q[0] + 1.0) < TOLERANCE:
            return np.array([0.0, np.pi, 0.0, 0.0]) # Or (0, 0, pi, 0), etc.
        # Otherwise, it's a non-unit scalar, which shouldn't happen for unit quaternions
        return np.array([np.log(q[0]), 0.0, 0.0, 0.0]) # This branch is mostly for non-unit scalar quaternions

    # Standard case for a unit quaternion q = [cos(theta), v*sin(theta)]
    # log(q) = [0, v*theta]
    v_norm = np.linalg.norm(q[1:])
    v = q[1:] / v_norm
    theta = np.arccos(np.clip(q[0], -1.0, 1.0)) # Clip to avoid domain errors for arccos

    return vec_to_quat(v * theta)

def quat_exp(q):
    """Calculates the exponential of a pure quaternion."""
    # For a pure quaternion q = [0, v], exp(q) = [cos(|v|), (v/|v|)sin(|v|)]
    if abs(q[0]) > TOLERANCE:
        # If the real part is not zero, it's not a pure quaternion.
        # For numerical stability, if it's very small, treat as zero.
        if abs(q[0]) < TOLERANCE:
            q[0] = 0.0
        else:
            raise ValueError(f"Input to quat_exp must be a pure quaternion (real part close to 0). Got: {q[0]}")

    v = q[1:]
    v_norm = np.linalg.norm(v)

    if v_norm < TOLERANCE:
        return np.array([1.0, 0.0, 0.0, 0.0])

    w = np.cos(v_norm)
    vec_part = (v / v_norm) * np.sin(v_norm)

    return np.array([w, vec_part[0], vec_part[1], vec_part[2]])


