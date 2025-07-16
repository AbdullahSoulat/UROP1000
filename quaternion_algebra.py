import numpy as np
# A tolerance for floating point comparisons
TOLERANCE = 1e-8

def vec_to_quat(v):
    # v: 3D unit vector
    north = np.array([0.0, 0.0, 1.0])
    if np.allclose(v, north):
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    if np.allclose(v, -north):
        return np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x axis
    axis = np.cross(north, v)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(north, v), -1.0, 1.0))
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.concatenate(([w], xyz))

def quat_to_vec(q):
    # q: quaternion as [w, x, y, z]
    # Rotate north pole by q
    north = np.array([0.0, 0.0, 1.0])
    # Convert north to pure quaternion
    v_quat = np.concatenate(([0.0], north))
    # q * v_quat * q_conjugate
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    temp = quat_multiply(q, v_quat)
    rotated = quat_multiply(temp, q_conj)
    return rotated[1:]  # Return the vector part

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
    """
    Calculates the inverse of a quaternion q = [w, x, y, z].
    For any quaternion, the inverse is the conjugate divided by the squared norm.
    """
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    norm_sq = np.dot(q, q)
    if norm_sq < TOLERANCE:
        raise ZeroDivisionError("Cannot invert a quaternion with near-zero norm.")
    return q_conj / norm_sq

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

    return np.concatenate(([0.0], v * theta))

def quat_exp(q):
    """
    Calculates the exponential of a quaternion q = [a, u].
    Returns exp(q) = exp(a) * [cos(|u|), (sin(|u|)/|u|) * u]
    """
    a = q[0]
    u = q[1:]
    u_norm = np.linalg.norm(u)

    exp_a = np.exp(a)
    if u_norm < TOLERANCE:
        return np.array([exp_a, 0.0, 0.0, 0.0])

    w = exp_a * np.cos(u_norm)
    xyz = exp_a * (np.sin(u_norm) / u_norm) * u
    return np.concatenate(([w], xyz))


