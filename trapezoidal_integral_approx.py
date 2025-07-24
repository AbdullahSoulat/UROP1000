import time
import numpy as np

# Define the function f(x)
def f(x):
    return np.array([np.cos(x) * np.cos(2*x),
                     np.sin(x) * np.cos(2*x),
                     np.sin(2*x)])

# Trapezium Approximation Algorithm
def trapezium_approx(f, a, b, N):
    delta_x = (b - a) / N
    approx_integral = np.array([0.0, 0.0, 0.0])

    approx_integral += f(a) + f(b)
    for i in range(1, N):
        x_i = a + i * delta_x
        approx_integral += 2 * f(x_i)

    return approx_integral * (delta_x / 2)

# I_exact = np.array([1/3, -1/3, 1])
# 
# N_values = [1, 2, 4, 8, 16, 32, 62, 128, 256, 512, 1024]
# results = []
# 
# for N in N_values:
#     start_time = time.time()
#     I_approx = trapezium_approx(f, 0, np.pi/2, N)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
# 
#     error = np.linalg.norm(I_approx - I_exact)
#     results.append({'N': N, 'Approx_Integral': I_approx, 'Error': error, 'Time': elapsed_time})
# 
# print(f"Exact Integral: {I_exact}")
# print("-" * 50)
# 
# for res in results:
#     print(f"N = {res['N']:<4}: Approx = {res['Approx_Integral']}, Error = {res['Error']:.10f}, Time = {res['Time']:.10f}")
# 
