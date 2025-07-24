# importing libraries
import time
import numpy as np
from sympy import symbols, integrate, sin, cos
from matplotlib import pyplot as plt

# importing functions from other files
# from slerp_integral_approx import spherical_trapezium_rule
from slerp_intergral_approx import spherical_trapezium_rule
from trapezoidal_integral_approx import trapezium_approx
from simpson_integral_approx import squad_integral_approximation_simpson
from squad_integral_approx import squad_integral_approx


# Defining constants
I_exact = np.array([1/3, -1/3, 1])
N_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
a = 0
b = np.pi/2

# Calculating the exact integral using sympy
x = symbols('x')
f_x = cos(x) * cos(2*x)
f_y = sin(x) * cos(2*x)
f_z = sin(2*x)

integrate_f_x = float(integrate(f_x, (x, a, b)))
integrate_f_y = float(integrate(f_y, (x, a, b)))
integrate_f_z = float(integrate(f_z, (x, a, b)))

I_exact = np.array([integrate_f_x, integrate_f_y, integrate_f_z])

# Define the function f(x)
def f(x):
    return np.array([np.cos(x) * np.cos(2*x),
                     np.sin(x) * np.cos(2*x),
                     np.sin(2*x)])

def calculating_results(integration_method, array):
    for N in N_values:
        start_time = time.time()
        I_approx = integration_method(f, a, b, N)
        end_time = time.time()
        elapsed_time = end_time - start_time

        error = np.linalg.norm(I_approx - I_exact)
        array.append({'N': N, 'Approx_Integral': I_approx, 'Error': error, 'Time': elapsed_time})

def print_results(array):
    print(f"Exact Integral: {I_exact}")
    print("-"*50)
    for res in array:
        print(f"N = {res['N']:<4}: Approx = {res['Approx_Integral']}, Error = {res['Error']:.10f}, Time = {res['Time']:.10f}")
    


# Arrays for all numerical integration methods
trapezoidal_results = []
slerp_results = []
simpson_results = []
squad_results = []

# Calculating the results
calculating_results(spherical_trapezium_rule, slerp_results)
calculating_results(squad_integral_approx, squad_results)
calculating_results(trapezium_approx, trapezoidal_results)
calculating_results(squad_integral_approximation_simpson, simpson_results)

# Arrays for plotting
dx = [1 / (2**i) for i in range(1, 11)]
log_dx = np.log(dx)

# Error arrays
error_slerp = np.array([d["Error"] for d in slerp_results])
error_trapezoidal = np.array([d["Error"] for d in trapezoidal_results])
error_simpson = np.array([d["Error"] for d in simpson_results])
error_squad = np.array([d["Error"] for d in squad_results])

log_error_slerp = np.log(error_slerp)
log_error_trapezoidal = np.log(error_trapezoidal)
log_error_simpson = np.log(error_simpson)
log_error_squad = np.log(error_squad)

slerp_convergence, b = np.polyfit(log_dx, log_error_slerp, deg=1)
trapezoidal_convergence, b = np.polyfit(log_dx, log_error_trapezoidal, deg=1)
simpson_convergence, b = np.polyfit(log_dx, log_error_simpson, deg=1)
squad_convergence, b = np.polyfit(log_dx, log_error_squad, deg=1)

# Time arrays
time_slerp = np.array([t["Time"] for t in slerp_results])
time_trapezoidal = np.array([t["Time"] for t in trapezoidal_results])
time_simpson = np.array([t["Time"] for t in simpson_results])
time_squad = np.array([t["Time"] for t in squad_results])

log_time_slerp = np.log(time_slerp)
log_time_trapezoidal = np.log(time_trapezoidal)
log_time_simpson = np.log(time_simpson)
log_time_sqaud = np.log(time_squad)

# Printing the results in table form
print("SLERP Results")
print_results(slerp_results)
print(f"Order of convergence: {slerp_convergence}")
print("")
print("")

print("Trapezoidal Results")
print_results(trapezoidal_results)
print(f"Order of convergence: {trapezoidal_convergence}")
print("")
print("")

print("Simpson Results")
print_results(simpson_results)
print(f"Order of convergence: {simpson_convergence}")
print("")
print("")

print("SQUAD Results")
print_results(squad_results)
print(f"Order of convergence: {squad_convergence}")
print("")
print("")


# Plotting the results using matplotlib

# plt.scatter(log_dx, log_error_slerp)
# plt.scatter(log_dx, log_error_trapezoidal)
# plt.scatter(log_dx, log_error_simpson)
# plt.scatter(log_dx, log_error_squad)

plt.plot(log_dx, log_error_slerp, label="SLEPR")
plt.plot(log_dx, log_error_trapezoidal, label="Trapezoidal")
plt.plot(log_dx, log_error_simpson, label="Simpson")
plt.plot(log_dx, log_error_squad, label="SQUAD")

# Drawing the line of best fit
em, b = np.polyfit(log_dx, log_error_slerp, deg=1)
# plt.plot(log_dx, m * log_dx + b, color='red', label='best fit line')

plt.title("Numerical Integration Method on 2-Sphere Curves")
plt.xlabel('log_dx')
plt.ylabel('log_error')
plt.grid(True)
plt.legend()
plt.show()
