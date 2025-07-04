import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define SLERP function
def slerp(p, q, u):
    p = np.array(p)
    q = np.array(q)
    dot = np.dot(p, q)
    dot = np.clip(dot, -1.0, 1.0)  # Avoid numerical errors
    theta = np.arccos(dot)
    if np.abs(theta) < 1e-10:  # If points are very close
        return p
    sin_theta = np.sin(theta)
    return (np.sin((1-u)*theta)/sin_theta) * p + (np.sin(u*theta)/sin_theta) * q

# Sample function f: R -> S^2
def f(t):
    return np.array([np.cos(t) * np.cos(2*t), np.sin(t) * np.cos(2*t), np.sin(2*t)])

# Generate points for the sphere surface
def generate_sphere_surface():
    phi = np.linspace(0, np.pi, 50)  # Polar angle
    theta = np.linspace(0, 2 * np.pi, 50)  # Azimuthal angle
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

# Generate points for the curve and SLERP segments
def generate_curve_and_slerp(f, a, b, n, slerp_points=5):
    t = np.linspace(a, b, n+1)  # Grid points
    points = [f(ti) for ti in t]
    slerp_curves = []
    for i in range(n):
        u = np.linspace(0, 1, slerp_points)
        segment = [slerp(points[i], points[i+1], ui) for ui in u]
        slerp_curves.append(np.array(segment))
    return points, slerp_curves, t

# Plotting function
def plot_sphere_and_curve():
    # Parameters
    a, b = 0, np.pi  # Interval for t
    n = 10  # Number of subintervals for SLERP
    slerp_points = 20  # Points per SLERP segment

    # Generate data
    x_sphere, y_sphere, z_sphere = generate_sphere_surface()
    points, slerp_curves, t = generate_curve_and_slerp(f, a, b, n, slerp_points)
    curve = np.array([f(ti) for ti in np.linspace(a, b, 100)])  # Smooth curve for reference

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere (semi-transparent)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.1, rstride=1, cstride=1)

    # Plot the smooth curve
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'r-', label='Curve f(t)', linewidth=2)

    # Plot discrete points
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='k', s=50, label='Sample points')

    # Plot SLERP segments
    for segment in slerp_curves:
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], 'g--', label='SLERP segments' if segment is slerp_curves[0] else "")

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Curve on 2-Sphere with SLERP Approximation')
    ax.legend()

    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.show()

# Run the visualization
if __name__ == "__main__":
    plot_sphere_and_curve()
