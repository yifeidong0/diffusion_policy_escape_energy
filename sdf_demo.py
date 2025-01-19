import numpy as np
import matplotlib.pyplot as plt

def compute_sdf_and_gradient(circles, grid_size=100):
    """
    Compute the Signed Distance Function (SDF) and its gradient for a set of circular obstacles in 2D space.

    Args:
        circles: List of tuples representing circle centers (x, y) and radii (r).
                 Example: [(x1, y1, r1), (x2, y2, r2), ...]
        grid_size: The resolution of the grid for SDF calculation.

    Returns:
        sdf_grid: 2D numpy array representing the SDF values on the grid.
        grad_x: 2D numpy array representing the x-component of the gradient.
        grad_y: 2D numpy array representing the y-component of the gradient.
    """
    # Create a grid of points within the unit square [0, 1] x [0, 1]
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # Initialize the SDF grid and gradient components
    sdf_grid = np.full(X.shape, np.inf)  # Set all values initially to infinity
    grad_x = np.zeros(X.shape)
    grad_y = np.zeros(Y.shape)

    # Iterate over each circle and compute the SDF and gradient
    min_dist = np.full(X.shape, np.inf)
    for cx, cy, r in circles:
        # Calculate the distance from each point in the grid to the circle center
        dist = np.linalg.norm(points - np.array([cx, cy]), axis=1) - r
        sdf_grid = np.minimum(sdf_grid, dist.reshape(X.shape))
        min_dist = np.minimum(min_dist, dist.reshape(X.shape))

        # Compute gradient of the distance to the circle center
        dist_x = X - cx
        dist_y = Y - cy
        norm = np.sqrt(dist_x**2 + dist_y**2)
        
        # Avoid division by zero for points at the circle center (though these should not exist in our grid)
        norm[norm == 0] = np.inf
        grad_x = np.where(min_dist == dist.reshape(X.shape), dist_x / norm, grad_x)
        grad_y = np.where(min_dist == dist.reshape(X.shape), dist_y / norm, grad_y)

    return sdf_grid, grad_x, grad_y

# Example usage:
# Define circle obstacles as (x_center, y_center, radius)
circles = [(0.3, 0.3, 0.1), (0.7, 0.7, 0.1), (0.5, 0.2, 0.15)]

# Compute the SDF and its gradient for the circles
sdf_grid, grad_x, grad_y = compute_sdf_and_gradient(circles, grid_size=25)

# Plot the SDF
plt.imshow(sdf_grid, extent=[0, 1, 0, 1], origin='lower', cmap='Spectral')

# Plot the circles
for cx, cy, r in circles:
    circle = plt.Circle((cx, cy), r, color='black', fill=False)
    plt.gca().add_artist(circle)

# Plot where the SDF is zero
plt.contour(sdf_grid, levels=[0], extent=[0, 1, 0, 1], colors='black', linestyles='dashed')

# Plot the gradient vectors as arrows
plt.quiver(np.linspace(0, 1, 25), np.linspace(0, 1, 25), grad_x, grad_y, color='black', scale=50, alpha=0.5)

plt.colorbar(label="Signed Distance")
plt.title("SDF and Gradient (Arrows) for Circular Obstacles")
plt.show()
