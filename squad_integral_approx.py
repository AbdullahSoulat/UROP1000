import time
import numpy as np
import quaternion_algebra as qa # Assuming quaternion_algebra.py is in the same directory

# A tolerance for floating point comparisons
TOLERANCE = 1e-8

def slerp(q0, q1, t):
    """
    Performs Spherical Linear Interpolation (SLERP) between two unit quaternions.

    Args:
        q0 (np.array): The starting unit quaternion.
        q1 (np.array): The ending unit quaternion.
        t (float): The interpolation parameter, typically between 0 and 1.

    Returns:
        np.array: The interpolated unit quaternion.
    """
    # Ensure quaternions are unit quaternions
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    # Ensure shortest path by making dot product positive
    dot_product = np.dot(q0, q1)
    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product

    # Clamp dot product to avoid domain errors for arccos due to floating point inaccuracies
    dot_product = np.clip(dot_product, -1.0, 1.0)

    theta = np.arccos(dot_product)

    if abs(theta) < TOLERANCE:
        # If the angle is very small, q0 and q1 are almost identical,
        # so just return q0 (or q1) to avoid division by zero.
        return q0
    
    sin_theta = np.sin(theta)
    
    # Handle the case where sin_theta is very small (theta close to 0 or pi)
    if abs(sin_theta) < TOLERANCE:
        # If theta is close to 0, return q0. If theta is close to pi,
        # it's an antipodal case, and linear interpolation is often used
        # as SLERP becomes ill-defined. For simplicity here, we return q0.
        return q0

    # SLERP formula
    return (q0 * np.sin((1 - t) * theta) + q1 * np.sin(t * theta)) / sin_theta


def generate_intermediate_squad_quaternion(q_prev, q_curr, q_next):
    """
    Generates an intermediate control quaternion (s_i) for SQUAD interpolation,
    also known as a "squad tangent" or "squad point".

    This function calculates the quaternion `s_i` such that the curve passes smoothly
    through `q_curr` while taking into account `q_prev` and `q_next`.

    The formula used is:
    s_i = q_i * exp(-1/4 * (log(q_i_inv * q_i_plus_1) + log(q_i_inv * q_i_minus_1)))

    Args:
        q_prev (np.array): The previous quaternion (q_{i-1}).
        q_curr (np.array): The current quaternion (q_i).
        q_next (np.array): The next quaternion (q_{i+1}).

    Returns:
        np.array: The intermediate control quaternion s_i.
    """
    # Ensure quaternions are unit quaternions
    q_prev = q_prev / np.linalg.norm(q_prev)
    q_curr = q_curr / np.linalg.norm(q_curr)
    q_next = q_next / np.linalg.norm(q_next)

    # Ensure shortest path for neighboring quaternions
    if np.dot(q_curr, q_prev) < 0:
        q_prev = -q_prev
    if np.dot(q_curr, q_next) < 0:
        q_next = -q_next

    # Calculate inverse of current quaternion
    q_curr_inv = qa.quat_inverse(q_curr)

    # Calculate log terms
    log_term1 = qa.quat_log(qa.quat_multiply(q_curr_inv, q_next))
    log_term2 = qa.quat_log(qa.quat_multiply(q_curr_inv, q_prev))

    # Sum the log terms
    log_sum = log_term1 + log_term2

    # Multiply by -1/4 and take the exponential
    exp_term = qa.quat_exp(-0.25 * log_sum)

    # Multiply q_curr by the exponential term to get s_i
    s_i = qa.quat_multiply(q_curr, exp_term)
    
    # Normalize s_i to ensure it's a unit quaternion
    return s_i / np.linalg.norm(s_i)


def squad(q0, q1, s0, s1, t):
    """
    Performs Spherical Cubic Interpolation (SQUAD) between two unit quaternions
    q0 and q1, using intermediate control quaternions s0 and s1.

    SQUAD is a double SLERP: SLERP(SLERP(q0, q1, t), SLERP(s0, s1, t), 2t(1-t)).

    Args:
        q0 (np.array): The starting unit quaternion (q_i).
        q1 (np.array): The ending unit quaternion (q_{i+1}).
        s0 (np.array): The intermediate control quaternion for q0 (s_i).
        s1 (np.array): The intermediate control quaternion for q1 (s_{i+1}).
        t (float): The interpolation parameter, typically between 0 and 1.

    Returns:
        np.array: The interpolated unit quaternion.
    """
    # First level of SLERP
    slerp_q0_q1 = slerp(q0, q1, t)
    slerp_s0_s1 = slerp(s0, s1, t)

    # Second level of SLERP with the blending parameter 2t(1-t)
    # This parameter ensures that the curve passes through q0 at t=0 and q1 at t=1
    # with the correct tangents.
    blending_param = 2 * t * (1 - t)
    
    return slerp(slerp_q0_q1, slerp_s0_s1, blending_param)

def f(x):
    return np.array([np.cos(x) * np.cos(2*x),
                     np.sin(x) * np.cos(2*x),
                     np.sin(2*x)])

# def squad_integral_approx(f, a, b, N):
#     """
#     Approximates the integral of a sphere-valued function using SQUAD interpolation and the trapezoidal rule.
#     Args:
#         f (function): A function f(t) that returns a 3D numpy array on the unit sphere.
#         a (float): The start of the integration interval.
#         b (float): The end of the integration interval.
#         N (int): The number of subintervals (keyframes) to use.
#     Returns:
#         numpy.ndarray: The 3D vector result of the integral approximation.
#     """
#     # 1. Sample the function at N+1 points
#     t_points = np.linspace(a, b, N + 1)
#     q_keyframes = np.array([qa.vec_to_quat(f(t)) for t in t_points])
# 
#     # 2. Compute SQUAD control points for each keyframe
#     s_control_points = np.zeros_like(q_keyframes)
#     for i in range(N + 1):
#         if i == 0:
#             # For the first keyframe, reflect the first segment
#             s_control_points[i] = generate_intermediate_squad_quaternion(q_keyframes[0], q_keyframes[0], q_keyframes[1])
#         elif i == N:
#             # For the last keyframe, reflect the last segment
#             s_control_points[i] = generate_intermediate_squad_quaternion(q_keyframes[N-1], q_keyframes[N], q_keyframes[N])
#         else:
#             s_control_points[i] = generate_intermediate_squad_quaternion(q_keyframes[i-1], q_keyframes[i], q_keyframes[i+1])
# 
#     # 3. Integrate over each interval using SQUAD and the trapezoidal rule
#     total_integral = np.zeros(3)
#     # sub_intervals = max(1000, N * 50)  # Fine sampling for accuracy
#     sub_intervals = 100
#     for i in range(N):
#         t0 = t_points[i]
#         t1 = t_points[i+1]
#         delta_t = (t1 - t0) / sub_intervals
#         interval_integral = np.zeros(3)
#         for j in range(sub_intervals + 1):
#             u = j / sub_intervals  # u in [0, 1]
#             quat_interp = squad(
#                 q_keyframes[i], q_keyframes[i+1],
#                 s_control_points[i], s_control_points[i+1],
#                 u
#             )
# 
#             # print(quat_interp) # printing for debugging
# 
#             vec_interp = qa.quat_to_vec(quat_interp)
# 
#             # Simpsons rule
#             # if j == 0 or j == sub_intervals:
#             #     interval_integral += vec_interp
#             # elif j % 2 == 1:
#             #     interval_integral += 4 * vec_interp
#             # else:
#             #     interval_integral += 2 * vec_interp
#             
#             # trapezoidal rule
#             weight = 1.0 if (j == 0 or j == sub_intervals) else 2.0 
#             interval_integral += weight * vec_interp
#         interval_integral *= delta_t / 2
#         total_integral += interval_integral
# 
#     return total_integral


def squad_integral_approx(f, a, b, N):
    """
    4th-order accurate integral approximation using SQUAD with proper endpoint handling.
    """
    # Sample function with one extra point at each end for better boundary handling
    t_points = np.linspace(a, b, N + 1)
    extended_t = np.concatenate(([a - (b-a)/N], t_points, [b + (b-a)/N]))
    q_extended = np.array([qa.vec_to_quat(f(t)) for t in extended_t])
    
    # Compute control points using central differences everywhere
    s_control_points = []
    for i in range(1, len(q_extended)-1):
        s_i = generate_intermediate_squad_quaternion(
            q_extended[i-1], q_extended[i], q_extended[i+1])
        s_control_points.append(s_i)
    s_control_points = np.array(s_control_points)
    
    # Use 4th-order Simpson's rule adapted for SQUAD
    total_integral = np.zeros(3)
    for i in range(N):
        t0 = t_points[i]
        t1 = t_points[i+1]
        dt = t1 - t0
        
        # Sample at t0, midpoint, and t1
        quat0 = q_extended[i+1]  # +1 because of extended array
        quat1 = q_extended[i+2]
        s0 = s_control_points[i]
        s1 = s_control_points[i+1]
        
        # Evaluate at three Simpson points
        vec0 = qa.quat_to_vec(squad(quat0, quat1, s0, s1, 0.0))
        vec_mid = qa.quat_to_vec(squad(quat0, quat1, s0, s1, 0.5))
        vec1 = qa.quat_to_vec(squad(quat0, quat1, s0, s1, 1.0))
        
        # Simpson's rule
        total_integral += dt * (vec0 + 4*vec_mid + vec1) / 6
    
    return total_integral


if __name__ == "__main__":
    I_exact = np.array([1/3, -1/3, 1])
    N_values = [1, 2, 4, 8, 16, 32, 62, 128, 256, 512, 1024]
    results = []

    for N in N_values:
        start_time = time.time()
        I_approx = squad_integral_approx(f, 0, np.pi/2, N)
        end_time = time.time()
        elapsed_time = end_time - start_time

        error = np.linalg.norm(I_approx - I_exact)
        results.append({'N': N, 'Approx_Integral': I_approx, 'Error': error, 'Time': elapsed_time})

    print(f"Exact Integral: {I_exact}")
    print("-" * 50)
    for res in results:
        print(f"N = {res['N']:<4}: Approx = {res['Approx_Integral']}, Error = {res['Error']:.10f}, Time = {res['Time']:.10f}")

    errors = [res['Error'] for res in results]
    Ns = [res['N'] for res in results]
    log_errors = np.log(errors)
    log_Ns = np.log(Ns)
    
    # Fit a line to find convergence order
    coefficients = np.polyfit(log_Ns, log_errors, 1)
    print(f"Convergence order: {-coefficients[0]}")
