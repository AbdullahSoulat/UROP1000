import time
from quaternion_algebra import *

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

    # 1. Define the time points for Simpson's Rule
    t_simpson_points = np.linspace(a, b, num_simpson_intervals + 1)
    h = (b - a) / num_simpson_intervals 

    # 2. Sample the function p_func at these points to get initial keyframe quaternions
    q_keyframes = np.array([vec_to_quat(p_func(t)) for t in t_simpson_points])
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

    exact_integral = np.array([1.0/3.0, -1.0/3.0, 1.0])

    # Format and print the header
    exact_str = np.array2string(exact_integral, precision=8, separator=' ', suppress_small=True)
    print(f"Exact Integral: {exact_str.replace('[', '[ ').replace(']', ' ]')}")
    print("-" * 70)

    # Loop through different numbers of intervals (N must be even for Simpson's rule)
    # Start from 2 intervals (2**1) up to 1024 (2**10)
    for i in range(1, 11): # N = 2, 4, 8, ..., 1024
        num_intervals = 2**i
        
        start_time = time.time()
        try:
            result_vector = squad_integral_approximation_simpson(
                p_func=specific_curve,
                a=start_interval,
                b=end_interval,
                num_simpson_intervals=num_intervals
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            error = np.linalg.norm(result_vector - exact_integral)

            # Format the output string to match the request as closely as possible
            approx_str = np.array2string(result_vector, precision=8, separator=' ', suppress_small=True)
            approx_str = approx_str.replace('[', '[ ').replace(']', ' ]')

            print(f"N = {num_intervals:<4}: Approx = {approx_str}, Error = {error:.10f}, Time = {elapsed_time:.10f}")

        except ValueError as e:
            print(f"N = {num_intervals:<4}: Error - {e}")
