import numpy as np

# Define the function f(x)
def f(x):
    return np.array([np.cos(x) * np.cos(2*x),
                     np.sin(x) * np.cos(2*x),
                     np.sin(2*x)])

# Spherical Trapezium Rule implementation
def spherical_trapezium_rule(f, a, b, N):
    delta_x = (b - a) / N
    approx_integral = np.array([0.0, 0.0, 0.0]) # Initialize as a 3D vector

    for i in range(N):
        x_i = a + i * delta_x
        x_i_plus_1 = a + (i + 1) * delta_x

        P_i = f(x_i)
        P_i_plus_1 = f(x_i_plus_1)

        # Ensure points are unit vectors (important for numerical stability of arccos)
        P_i = P_i / np.linalg.norm(P_i)
        P_i_plus_1 = P_i_plus_1 / np.linalg.norm(P_i_plus_1)

        dot_product = np.dot(P_i, P_i_plus_1)
        
        # Clamp dot product to handle floating point errors that might push it slightly outside [-1, 1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        theta_i = np.arccos(dot_product)

        S_i = 0.0
        epsilon = 1e-9 # Threshold for small angles
        if abs(theta_i) < epsilon:
            S_i = 0.5
        elif abs(theta_i - np.pi) < epsilon: # Handle antipodal case, although unlikely for continuous functions
            # This is tricky. For true antipodal, SLERP is not unique.
            # Numerically, if theta_i is very close to pi, sin(theta_i) is very small.
            # (1 - cos(theta_i)) / (theta_i * sin(theta_i))
            # cos(pi) = -1, so 1 - cos(pi) = 2.
            # If theta_i approaches pi, the term approaches 2 / (pi * very_small_number) -> Inf
            # This indicates an issue with the SLERP path choice for antipodal points.
            # For practical numerical integration, if your function is smooth, you'd usually
            # expect theta_i not to hit exactly pi unless N is extremely small.
            # We'll let it compute, and note if it leads to large errors.
            # A more robust handling for antipodal points might involve a different integration scheme
            # or a warning/error, as the SLERP assumption breaks down.
            S_i = (1 - np.cos(theta_i)) / (theta_i * np.sin(theta_i))
        else:
            S_i = (1 - np.cos(theta_i)) / (theta_i * np.sin(theta_i))
        
        # Accumulate the contribution
        approx_integral += delta_x * S_i * (P_i + P_i_plus_1)

    return approx_integral

# True integral value
I_exact = np.array([1/3, -1/3, 1])

# Test with various N values
N_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
results = []

for N in N_values:
    I_approx = spherical_trapezium_rule(f, 0, np.pi/2, N)
    error = np.linalg.norm(I_approx - I_exact)
    results.append({'N': N, 'Approx_Integral': I_approx, 'Error': error})

print(f"Exact Integral: {I_exact}")
print("-" * 50)
for res in results:
    print(f"N = {res['N']:<4}: Approx = {res['Approx_Integral']}, Error = {res['Error']:.10f}")
