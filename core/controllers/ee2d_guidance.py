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
        self.cost_weights = np.array([0., 1e-4, 0.0, 0.]) # TODO: from config
        cost_gradients = -(self.cost_weights[0] * self._potential_func_dot(pred_action)
                           + self.cost_weights[1] * self._collision_avoidance_func_dot(obstacle_info, pred_action)
                           + self.cost_weights[2] * self._path_length_func_dot(pred_action)
                           + self.cost_weights[3] * self._goal_reach_func_dot(obstacle_info, pred_action))
        # cost_gradients = np.zeros(pred_action.shape)
        return cost_gradients

    # @staticmethod
    # def _collision_avoidance_func(obstacle_info, pred_action) -> np.ndarray:
    #     costs = np.zeros(pred_action.shape[0])
    #     for k, act in enumerate(pred_action.tolist()):
    #         cost = 0
    #         for obs_radius, obs_center in zip(obstacle_info["radius"], obstacle_info["center"]):
    #             # each obstacle is an ellipse
    #             cost += 1 - (act[0] - obs_center[0]) ** 2 / obs_radius[0] ** 2 - (act[1] - obs_center[1]) ** 2 / obs_radius[1] ** 2
    #         costs[k] = cost
    #     return costs
                
    # @staticmethod
    # def _collision_avoidance_func_dot(obstacle_info, pred_action) -> np.ndarray:
    #     cost_grads = np.zeros(pred_action.shape)
    #     for k, act in enumerate(pred_action.tolist()):
    #         grad = np.zeros(pred_action.shape[1])
    #         for obs_radius, obs_center in zip(obstacle_info["radius"], obstacle_info["center"]):
    #             # each obstacle is an ellipse
    #             dist_x = (act[0] - obs_center[0]) ** 2 / obs_radius[0] ** 2
    #             dist_y = (act[1] - obs_center[1]) ** 2 / obs_radius[1] ** 2
                
    #             # Compute gradient only if inside the ellipse (dist_x + dist_y <= 1)
    #             if dist_x + dist_y <= 1:
    #                 grad[0] += -2 * (act[0] - obs_center[0]) / obs_radius[0] ** 2
    #                 grad[1] += -2 * (act[1] - obs_center[1]) / obs_radius[1] ** 2
    #             else:
    #                 grad[0] += 0  # Outside the ellipse, no gradient contribution
    #                 grad[1] += 0

    #         cost_grads[k] = grad
    #     return cost_grads

    def _collision_avoidance_func_dot(self, obstacle_info, pred_action) -> np.ndarray:
        cost_grads = np.zeros(pred_action.shape)
        
        # Iterate through each waypoint and compute gradient for each
        for k in range(pred_action.shape[0] - 1):
            print('!!!!!!!!!!!k,',k,)
            act1 = pred_action[k]
            act2 = pred_action[k + 1]
            grad = np.zeros((2, pred_action.shape[1]))

            for obs_radius, obs_center in zip(obstacle_info["radius"], obstacle_info["center"]):
                # Gradients along the line segment between act1 and act2
                # print('!!!!!!!!!!!obs,', obs_radius, obs_center)
                grad += self._line_segment_collision_grad(act1, act2, obs_radius, obs_center)

            cost_grads[k:k+2] += grad

        # Compute gradient for the last waypoint
        for obs_radius, obs_center in zip(obstacle_info["radius"], obstacle_info["center"]):
            cost_grads[-1] += self._point_collision_grad(pred_action[-1], obs_radius, obs_center, t=1)[1,:]

        return cost_grads

    def _point_collision_grad(self, act, obs_radius, obs_center, t) -> np.ndarray:
        # Compute gradient of the collision cost at a given point
        grad = np.zeros((2,2))
        dist_x = (act[0] - obs_center[0]) ** 2 / obs_radius[0] ** 2
        dist_y = (act[1] - obs_center[1]) ** 2 / obs_radius[1] ** 2
        if dist_x + dist_y <= 1:
            print('!!!!!!!!!!!1-dist_x - dist_y,',1-dist_x - dist_y,)
            grad[0,0] = -2 * (1-t) * (act[0] - obs_center[0]) / obs_radius[0] ** 2
            grad[0,1] = -2 * (1-t) * (act[1] - obs_center[1]) / obs_radius[1] ** 2
            grad[1,0] = -2 * t * (act[0] - obs_center[0]) / obs_radius[0] ** 2
            grad[1,1] = -2 * t * (act[1] - obs_center[1]) / obs_radius[1] ** 2
            print('!!!!!!!!!!!grad,',grad)
        return grad

    def _line_segment_collision_grad(self, act1, act2, obs_radius, obs_center, num_samples=5) -> np.ndarray:
        # Gradients for points along the line segment
        grad = np.zeros((2,2))
        for t in np.linspace(0, 1, num_samples, endpoint=False):
            print('!!!!!!!!!!!t,',t,)
            point = (1 - t) * act1 + t * act2
            grad += self._point_collision_grad(point, obs_radius, obs_center, t)
        return grad
    
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

    @staticmethod
    def _goal_reach_func_dot(obstacle_info, pred_action) -> np.ndarray:
        grads = np.zeros(pred_action.shape)
        grads[-1,1] = 2 * (pred_action[-1,1]-obstacle_info["center"][1][1])
        return grads
    
    def set_config(self, config: Dict):
        self.cbf_alpha = config["cbf_clf_controller"]["cbf_alpha"]
        # self.clf_gamma = config["cbf_clf_controller"]["clf_gamma"]
        # self.penalty_slack_cbf = config["cbf_clf_controller"]["penalty_slack_cbf"]
        # self.penalty_slack_clf = config["cbf_clf_controller"]["penalty_slack_clf"]
        self.denoising_guidance_step = config["cbf_clf_controller"]["denoising_guidance_step"]
        # self.quadrotor_params = config["simulator"]

    # @staticmethod
    # def _barrier_func(y, z, obs_y, obs_z, obs_r) -> float:
    #     return (y - obs_y) ** 2 + (z - obs_z) ** 2 - (obs_r) ** 2

    # @staticmethod
    # def _barrier_func_dot(y, z, obs_y, obs_z) -> list:
    #     return [2 * (y - obs_y), 2 * (z - obs_z)]

    # @staticmethod
    # def _lyapunoc_func(y, z, des_y, des_z) -> float:
    #     return (y - des_y) ** 2 + (z - des_z) ** 2

    # @staticmethod
    # def _lyapunov_func_dot(y, z, des_y, des_z) -> list:
    #     return [2 * (y - des_y), 2 * (z - des_z)]

    # @staticmethod
    # def _define_QP_problem_data(
    #     u1: float,
    #     u2: float,
    #     cbf_alpha: float,
    #     clf_gamma: float,
    #     penalty_slack_cbf: float,
    #     penalty_slack_clf: float,
    #     h: list,
    #     coeffs_dhdx: list,
    #     v: list,
    #     coeffs_dvdx: list,
    #     vmin=-15.0,
    #     vmax=15.0,
    # ):
    #     vmin, vmax = -15.0, 15.0

    #     P = sparse.csc_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, penalty_slack_cbf, 0], [0, 0, 0, penalty_slack_clf]])
    #     q = np.array([-u1, -u2, 0, 0])
    #     A = sparse.csc_matrix(
    #         [c for c in coeffs_dhdx]
    #         + [c for c in coeffs_dvdx]
    #         + [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    #     )
    #     lb = np.array([-cbf_alpha * h_ for h_ in h] + [-np.inf for _ in v] + [vmin, vmin, 0, 0])
    #     ub = np.array([np.inf for _ in h] + [-clf_gamma * v_ for v_ in v] + [vmax, vmax, np.inf, np.inf])
    #     return P, q, A, lb, ub

    # @staticmethod
    # def _get_quadrotor_state(state):
    #     y, y_dot, z, z_dot, phi, phi_dot = state
    #     return y, y_dot, z, z_dot, phi, phi_dot

    # def _calculate_cbf_coeffs(self, state: np.ndarray, obs_center: List, obs_radius: List, minimal_distance: float):
    #     """
    #     Let barrier function be h and system state x, the CBF constraint
    #     h_dot(x) >= - alpha * h + δ
    #     """
    #     h = []  # barrier values (here, remaining distance to each obstacle)
    #     coeffs_dhdx = []  # dhdt = dhdx * dxdt = dhdx * u
    #     for center, radius in zip(obs_center, obs_radius):
    #         y, _, z, _, _, _ = self._get_quadrotor_state(state)
    #         h.append(self._barrier_func(y, z, center[0], center[1], radius + minimal_distance))
    #         # Additional [1, 0] incorporates the CBF slack variable into the constraint
    #         coeffs_dhdx.append(self._barrier_func_dot(y, z, center[0], center[1]) + [1, 0])
    #     return h, coeffs_dhdx

    # def _calculate_clf_coeffs(self, state: np.ndarray, target_y: float, _target_z: float):
    #     """
    #     Let Lyapunov function be v and system state x, the CBF constraint
    #     v_dot(x) - δ <= - gamma * v
    #     """
    #     y, _, z, _, _, _ = self._get_quadrotor_state(state)
    #     v = [self._lyapunoc_func(y, z, target_y, _target_z)]
    #     # Additional [0, -1] incorporates the CLF slack variable into the constraint
    #     coeffs_dvdx = [self._lyapunov_func_dot(y, z, target_y, _target_z) + [0, -1]]
    #     return v, coeffs_dvdx

    # def compute_c(
    #     self,
    #     state: np.ndarray,
    #     control: np.ndarray,
    #     obs_center: List,
    #     obs_radius: List,
    #     cbf_alpha: float = 15.0,
    #     clf_gamma: float = 0.01,
    #     penalty_slack_cbf: float = 1e2,
    #     penalty_slack_clf: float = 1.0,
    #     target_position: tuple = (5.0, 5.0),
    # ):
    #     """
    #     Calculate the safe command by solveing the following optimization problem

    #                 minimize  || u - u_nom ||^2 + k * δ^2
    #                   u, δ
    #                 s.t.
    #                         h'(x) ≥ -𝛼 * h(x) - δ1
    #                         v'(x) ≤ -γ * v(x) + δ2
    #                         u_min ≤ u ≤ u_max
    #                             0 ≤ δ1,δ2 ≤ inf
    #     where
    #         u = [ux, uy] is the control input in x and y axis respectively.
    #         δ is the slack variable
    #         h(x) is the control barrier function and h'(x) its derivative
    #         v(x) is the lyapunov function and v'(x) its derivative

    #     The problem above can be formulated as QP (ref: https://osqp.org/docs/solver/index.html)

    #                 minimize 1/2 * x^T * Px + q^T x
    #                     x
    #                 s.t.
    #                             l ≤ Ax ≤ u
    #     where
    #         x = [ux, uy, δ1, δ2]

    #     """
    #     u1, u2 = control
    #     target_y, target_z = target_position

    #     # Calculate values of the barrier function and coeffs in h_dot to state
    #     h, coeffs_dhdx = self._calculate_cbf_coeffs(state, obs_center, obs_radius, self.quadrotor_params["l_q"])
    #     # Calculate value of the lyapunov function and coeffs in v_dot to state
    #     v, coeffs_dvdx = self._calculate_clf_coeffs(state, target_y, target_z)

    #     # Define problem
    #     P, q, A, lb, ub = self._define_QP_problem_data(
    #         u1, u2, cbf_alpha, clf_gamma, penalty_slack_cbf, penalty_slack_clf, h, coeffs_dhdx, v, coeffs_dvdx
    #     )

    #     # Solve QP
    #     prob = osqp.OSQP()
    #     prob.setup(P, q, A, lb, ub, verbose=False, time_limit=0)
    #     # Solve QP problem
    #     res = prob.solve()

    #     safe_u1, safe_u2, _, _ = res.x
    #     return np.array([safe_u1, safe_u2])
