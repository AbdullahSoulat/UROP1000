
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

def slerp(q_a, q_b, tau):
    """Performs Spherical Linear Interpolation between two quaternions."""
    # Ensure inputs are unit quaternions
    q_a = q_a / np.linalg.norm(q_a)
    q_b = q_b / np.linalg.norm(q_b)

    # Calculate the dot product (cosine of the angle between them)
    dot = np.dot(q_a, q_b)

    # If the dot product is negative, the quaternions are more than 90 degrees
    # apart. slerp won't take the shorter path. So we flip one of the quaternions.
    if dot < 0.0:
        q_b = -q_b
        dot = -dot

    # If the quaternions are very close, use linear interpolation to avoid
    # division by zero issues.
    if dot > 1.0 - TOLERANCE:
        result = q_a + tau * (q_b - q_a)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)        # Angle between input quaternions
    theta = theta_0 * tau           # Angle for the result
    sin_theta = np.sin(theta)       # Computed now to avoid computing it twice
    sin_theta_0 = np.sin(theta_0)   # Computed now to avoid computing it twice

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q_a) + (s1 * q_b)


def squad_integral_approximation_simpson(p_func, a, b, num_simpson_intervals):
    """
    Approximates the integral of a sphere-valued function using SQUAD and Simpson's Rule.

    Args:
        p_func (function): A function p(t) that returns a 3D numpy array on the unit sphere.
        a (float): The start of the integration interval.
        b (float): The end of the integration interval.
        num_simpson_intervals (int): The number of subintervals to use for Simpson's rule.
                                     Must be an even number and at least 2.

    Returns:
        numpy.ndarray: The 3D vector result of the integral approximation.
    """
    if num_simpson_intervals < 2 or num_simpson_intervals % 2 != 0:
        raise ValueError("Number of subintervals for Simpson's Rule must be an even number >= 2.")

    # Helper function to calculate a SQUAD control point.
    # This is nested because its specific boundary handling for q_prev and q_next
    # depends on the context of the q_keyframes array.
    def calculateSquadControlPoint(q_prev, q_curr, q_next):
        # Ensure shortest path for dot products before log to avoid issues with antipodal quaternions
        q_prev_adj = q_prev
        if np.dot(q_prev, q_curr) < 0:
            q_prev_adj = -q_prev

        q_next_adj = q_next
        if np.dot(q_next, q_curr) < 0:
            q_next_adj = -q_next

        q_curr_inv = quat_inverse(q_curr) # q_curr^-1

        # Calculate log(q_curr^-1 * q_prev) and log(q_curr^-1 * q_next)
        log1 = quat_log(quat_multiply(q_curr_inv, q_prev_adj))
        log2 = quat_log(quat_multiply(q_curr_inv, q_next_adj))

        # Sum the logarithms
        sumLogs = log1 + log2 # Quaternion addition is just vector addition

        # Scale the sum by -0.25 (as per SQUAD formula)
        scaledSumLogs = -0.25 * sumLogs

        # Exponentiate the scaled sum
        expResult = quat_exp(scaledSumLogs)

        # Final control point calculation: q_curr * exp(...)
        return quat_multiply(q_curr, expResult)

    # The SQUAD function itself, which will be the function we integrate
    def get_squad_interpolated_point(q_keyframes, s_control_points, interval_idx, u):
        """
        Calculates a point on the SQUAD curve.
        Args:
            q_keyframes (np.ndarray): Array of keyframe quaternions.
            s_control_points (np.ndarray): Array of SQUAD control point quaternions.
            interval_idx (int): The index of the keyframe interval (0 to num_keyframes - 2).
            u (float): Interpolation parameter within the interval [0, 1].
        Returns:
            np.ndarray: The 3D vector on the unit sphere corresponding to the interpolated quaternion.
        """
        q_i = q_keyframes[interval_idx]
        q_i_plus_1 = q_keyframes[interval_idx + 1]
        s_i = s_control_points[interval_idx]
        s_i_plus_1 = s_control_points[interval_idx + 1]

        # Ensure slerp takes the shortest path by negating if dot product is negative
        # These adjustments are critical for correct SLERP behavior
        q_i_plus_1_adj = q_i_plus_1
        if np.dot(q_i_plus_1, q_i) < 0:
            q_i_plus_1_adj = -q_i_plus_1

        s_i_plus_1_adj = s_i_plus_1
        if np.dot(s_i_plus_1, s_i) < 0:
            s_i_plus_1_adj = -s_i_plus_1

        # First SLERP: Interpolate between keyframes
        slerp1 = slerp(q_i, q_i_plus_1_adj, u)
        # Second SLERP: Interpolate between control points
        slerp2 = slerp(s_i, s_i_plus_1_adj, u)

        # Final SLERP: Blend the two slerps using a cubic blending factor (2u(1-u))
        cubic_blend_factor = 2 * u * (1 - u)
        interpolated_quat = slerp(slerp1, slerp2, cubic_blend_factor)

        return quat_to_vec(interpolated_quat)


    # 1. Define the time points for Simpson's Rule
    t_simpson_points = np.linspace(a, b, num_simpson_intervals + 1)
    h = (b - a) / num_simpson_intervals # Step size for Simpson's rule

    # 2. Sample the function p_func at these points to get initial keyframe quaternions
    # These keyframes will be used to define the SQUAD curve.
    q_keyframes = np.array([vec_to_quat(p_func(t)) for t in t_simpson_points])

    # 3. Compute the control points s_i for the SQUAD curve
    # The number of control points will be the same as keyframes.
    s_control_points = np.zeros_like(q_keyframes)

    # Calculate all control points upfront.
    # Boundary conditions for first and last control points (common approximation):
    # For q_0, use q_0 and q_1.
    # For q_N, use q_{N-1} and q_N.
    # This approach effectively "reflects" the first/last segment.
    for i in range(q_keyframes.shape[0]):
        q_prev, q_curr, q_next = None, None, None
        if i == 0:
            q_prev = q_keyframes[0] # Treat q_0 as if it was preceded by itself for tangent calculation
            q_curr = q_keyframes[0]
            q_next = q_keyframes[1]
        elif i == q_keyframes.shape[0] - 1:
            q_prev = q_keyframes[i - 1]
            q_curr = q_keyframes[i]
            q_next = q_keyframes[i] # Treat q_N as if it was followed by itself
        else:
            q_prev = q_keyframes[i - 1]
            q_curr = q_keyframes[i]
            q_next = q_keyframes[i + 1]

        s_control_points[i] = calculateSquadControlPoint(q_prev, q_curr, q_next)

    # 4. Apply Simpson's Rule using the SQUAD-interpolated function values
    total_integral_vector = np.zeros(3)

    # Sum for Simpson's Rule
    # Simpson's Rule formula: (h/3) * [f(x_0) + 4f(x_1) + 2f(x_2) + ... + 4f(x_{N-1}) + f(x_N)]
    for i in range(num_simpson_intervals + 1):
        # The function value at t_simpson_points[i] is simply the vector part of q_keyframes[i],
        # since the SQUAD curve passes exactly through the keyframes.
        func_val = quat_to_vec(q_keyframes[i])

        if i == 0 or i == num_simpson_intervals:
            # First and last points have a weight of 1
            total_integral_vector += func_val
        elif i % 2 == 1:
            # Odd-indexed points have a weight of 4
            total_integral_vector += 4 * func_val
        else:
            # Even-indexed points have a weight of 2
            total_integral_vector += 2 * func_val

    # Final scaling by h/3
    return total_integral_vector * (h / 3.0)

# --- Example Usage ---
if __name__ == "__main__":
    # Define the function p(t) that traces a curve on the unit sphere
    def specific_curve(t):
        # This vector is already a unit vector, so no normalization is needed.
        return np.array([
            np.cos(t) * np.cos(2*t),
            np.sin(t) * np.cos(2*t),
            np.sin(2*t)
        ])

    # Define integration parameters
    start_interval = 0
    end_interval = np.pi / 2

    # The analytical integral of this function from 0 to pi/2 is [1/3, -1/3, 1].
    exact_integral = np.array([1.0/3.0, -1.0/3.0, 1.0])

    # Format and print the header
    exact_str = np.array2string(exact_integral, precision=8, separator=' ', suppress_small=True)
    print(f"Exact Integral: {exact_str.replace('[', '[ ').replace(']', ' ]')}")
    print("-" * 70)

    # Loop through different numbers of intervals (N must be even for Simpson's rule)
    # Start from 2 intervals (2**1) up to 1024 (2**10)
    for i in range(1, 11): # N = 2, 4, 8, ..., 1024
        num_intervals = 2**i

        try:
            result_vector = squad_integral_approximation_simpson(
                p_func=specific_curve,
                a=start_interval,
                b=end_interval,
                num_simpson_intervals=num_intervals
            )

            error = np.linalg.norm(result_vector - exact_integral)

            # Format the output string to match the request as closely as possible
            approx_str = np.array2string(result_vector, precision=8, separator=' ', suppress_small=True)
            approx_str = approx_str.replace('[', '[ ').replace(']', ' ]')

            print(f"N = {num_intervals:<4}: Approx = {approx_str}, Error = {error:.10f}")
        except ValueError as e:
            print(f"N = {num_intervals:<4}: Error - {e}")
