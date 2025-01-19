from typing import Dict, List
import numpy as np
import osqp
from scipy import sparse

from core.controllers.base_controller import BaseController


class EscapeEnergy2DGuidanceController(BaseController):
    """
    A CLF-CBF safety filter assuming a simple velocity-controled dynamics
        y_dot = u1
        z_dot = u2
    Barrier funciton h is defined as the distances to each obstacle
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        super().__init__(device)
        self.set_config(config)

    def compute_cost_gradients(self, obstacle_info: Dict[str, List[List]], pred_action: np.ndarray) -> np.ndarray:
        """
        pred_action: np.ndarray, shape=(pre_horizon, action_dim)
        Compute Line 10, Algorithm 1 in https://arxiv.org/abs/2308.01557
        """
        self.obstacle_info = obstacle_info
        self.num_obstacles = len(self.obstacle_info["center"])
        # self.cost_weights = np.array([0,0,0,])
        self.cost_weights = np.array([0.001, 1e-3, 0.001,]) # TODO: from config
        cost_gradients = -(self.cost_weights[0] * self._potential_func_dot(pred_action)
                           + self.cost_weights[1] * self._sdf_collision_avoidance_func_dot(obstacle_info, pred_action)
                           + self.cost_weights[2] * self._path_length_func_dot(pred_action))
                        #    + self.cost_weights[3] * self._goal_reach_func_dot(obstacle_info, pred_action))
        # cost_gradients = np.zeros(pred_action.shape)
        return cost_gradients

    def compute_sdf_and_gradient(self, circles, grid_size=100):
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

    def _sdf_collision_avoidance_func_dot(self, obstacle_info, pred_action) -> np.ndarray:
        cost_grads = np.zeros(pred_action.shape)
        
        # Compute the Signed Distance Function (SDF) and its gradient for the obstacles
        circles = list(np.hstack((obstacle_info["center"], obstacle_info["radius"])))
        sdf_grid, grad_x, grad_y = self.compute_sdf_and_gradient(circles, grid_size=100)
        
        # Iterate through each waypoint and compute gradient for each
        for k in range(pred_action.shape[0] - 1):
            act1 = pred_action[k]
            act2 = pred_action[k + 1]
            grad = self._sdf_collision_grad(act1, act2, sdf_grid, grad_x, grad_y)

            cost_grads[k:k+2] -= grad

        # Compute gradient for the last waypoint
        cost_grads[-1] -= self._sdf_point_collision_grad(pred_action[-1], sdf_grid, grad_x, grad_y)

        return cost_grads

    def _sdf_collision_grad(self, act1, act2, sdf_grid, grad_x, grad_y, epsilon=0):
        grad = np.zeros((2, 2))
        # Linearly interpolate between act1 and act2 to compute the gradient at each point
        for t in np.linspace(0, 1, 1, endpoint=False):  # Arbitrary 10 samples between the points
            point = (1 - t) * act1 + t * act2
            idx_x = max(0, min(sdf_grid.shape[0]-1, int(point[0] * sdf_grid.shape[0])))
            idx_y = max(0, min(sdf_grid.shape[1]-1, int(point[1] * sdf_grid.shape[1])))

            # Compute the gradient based on the SDF gradient
            if sdf_grid[idx_x, idx_y] < epsilon:
                sdf_gradient = np.array([[(1-t) * grad_x[idx_x, idx_y], (1-t) * grad_y[idx_x, idx_y]],
                                         [t * grad_x[idx_x, idx_y], t * grad_y[idx_x, idx_y]]])
                grad += sdf_gradient  # Add the gradient for each sampled point
        
        return grad

    def _sdf_point_collision_grad(self, act, sdf_grid, grad_x, grad_y, epsilon=0):
        # Compute the gradient of the collision cost at a given point
        idx_x = max(0, min(sdf_grid.shape[0]-1, int(act[0] * sdf_grid.shape[0])))
        idx_y = max(0, min(sdf_grid.shape[1]-1, int(act[1] * sdf_grid.shape[1])))

        if sdf_grid[idx_x, idx_y] < epsilon:
            return np.array([grad_x[idx_x, idx_y], grad_y[idx_x, idx_y]])
        else:
            return np.zeros(2)
    
    @staticmethod
    def _path_length_func_dot(pred_action) -> np.ndarray:
        grads = np.zeros(pred_action.shape)
        for k in range(1, pred_action.shape[0] - 1):
            grads[k] = 2 * (pred_action[k] - pred_action[k-1]) + 2 * (pred_action[k] - pred_action[k+1])
        grads[0] = 2 * (pred_action[0] - pred_action[1])
        grads[-1] = 2 * (pred_action[-1] - pred_action[-2])
        return grads

    @staticmethod
    def _potential_func(pred_action) -> float:
        return float(np.max(pred_action[:,1]) - pred_action[0,1])
    
    @staticmethod
    def _potential_func_dot(pred_action) -> np.ndarray:
        if np.argmax(pred_action[:,1]) == 0:
            return np.zeros(pred_action.shape)
        else:
            dcdtau = np.zeros(pred_action.shape)
            dcdtau[0,1] = -1
            dcdtau[np.argmax(pred_action[:,1]),1] = 1
            return dcdtau

    # @staticmethod
    # def _goal_reach_func_dot(obstacle_info, pred_action) -> np.ndarray:
    #     grads = np.zeros(pred_action.shape)
    #     grads[-1,1] = 2 * (pred_action[-1,1]-obstacle_info["center"][1][1])
    #     return grads
    
    def set_config(self, config: Dict):
        self.cbf_alpha = config["cbf_clf_controller"]["cbf_alpha"]
        # self.clf_gamma = config["cbf_clf_controller"]["clf_gamma"]
        # self.penalty_slack_cbf = config["cbf_clf_controller"]["penalty_slack_cbf"]
        # self.penalty_slack_clf = config["cbf_clf_controller"]["penalty_slack_clf"]
        self.denoising_guidance_step = config["cbf_clf_controller"]["denoising_guidance_step"]
        # self.quadrotor_params = config["simulator"]
