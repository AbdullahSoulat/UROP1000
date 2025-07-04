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

    for i in range(N):
        x_i = a + i * delta_x
        if (i == 0 or i == N):
            approx_integral += f(x_i) 
        else:
            approx_integral += 2 * f(x_i)

    return approx_integral * (delta_x / 2)

I_exact = np.array([1/3, -1/3, 1])

N_values = [1, 2, 4, 8, 16, 32, 62, 128, 256, 512, 1024]
results = []

for N in N_values:
    I_approx = trapezium_approx(f, 0, np.pi/2, N)
    error = np.linalg.norm(I_approx - I_exact)
    results.append({'N': N, 'Approx_Integral': I_approx, 'Error': error})

print(f"Exact Integral: {I_exact}")
print("-" * 50)

for res in results:
    print(f"N = {res['N']:<4}: Approx = {res['Approx_Integral']}, Error = {res['Error']:.10f}")

