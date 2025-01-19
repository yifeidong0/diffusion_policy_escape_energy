from typing import Dict, List

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

from core.controllers.ddpm_scheduler import DDPMSchedulerExtend
from core.controllers.base_controller import BaseController
from core.controllers.ee2d_guidance import EscapeEnergy2DGuidanceController
from core.networks.conditional_unet1d import ConditionalUnet1D
from utils.normalizers import BaseNormalizer


def build_networks_from_config(config: Dict):
    action_dim = config["controller"]["networks"]["action_dim"]
    obs_dim = config["controller"]["networks"]["obs_dim"]
    obs_horizon = config["obs_horizon"]
    obstacle_encode_dim = config["controller"]["networks"]["obstacle_encode_dim"]
    return ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon + obstacle_encode_dim)


def build_noise_scheduler_from_config(config: Dict):
    type_noise_scheduler = config["controller"]["noise_scheduler"]["type"]
    if type_noise_scheduler.lower() == "ddpm":
        return DDPMSchedulerExtend(
            num_train_timesteps=config["controller"]["noise_scheduler"]["ddpm"]["num_train_timesteps"],
            beta_schedule=config["controller"]["noise_scheduler"]["ddpm"]["beta_schedule"],
            clip_sample=config["controller"]["noise_scheduler"]["ddpm"]["clip_sample"],
            prediction_type=config["controller"]["noise_scheduler"]["ddpm"]["prediction_type"],
        )
    elif type_noise_scheduler.lower() == "ddim":
        return DDIMScheduler(
            num_train_timesteps=config["controller"]["noise_scheduler"]["ddim"]["num_train_timesteps"],
            beta_schedule=config["controller"]["noise_scheduler"]["ddim"]["beta_schedule"],
            clip_sample=config["controller"]["noise_scheduler"]["ddim"]["clip_sample"],
            prediction_type=config["controller"]["noise_scheduler"]["ddim"]["prediction_type"],
        )
    elif type_noise_scheduler.lower() == "dpmsolver":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=config["controller"]["noise_scheduler"]["dpmsolver"]["num_train_timesteps"],
            beta_schedule=config["controller"]["noise_scheduler"]["dpmsolver"]["beta_schedule"],
            prediction_type=config["controller"]["noise_scheduler"]["dpmsolver"]["prediction_type"],
            use_karras_sigmas=config["controller"]["noise_scheduler"]["dpmsolver"]["use_karras_sigmas"],
        )
    else:
        raise NotImplementedError


class EscapeEnergy2DDiffusionPolicy(BaseController):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMSchedulerExtend,
        normalizer: BaseNormalizer,
        ee2d_guidance: EscapeEnergy2DGuidanceController,
        config: Dict,
        device: str = "cuda",
    ):
        self.device = device
        self.net = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = normalizer

        self.set_config(config)
        self.net.to(self.device)

        self.ee2d_guidance = ee2d_guidance
        self.use_ee2d_guidance = False if ee2d_guidance is None else True

    def predict_action(self, obs_dict: Dict[str, List]) -> np.ndarray:
        # stack the observations
        obs_seq = np.stack(obs_dict["state"])
        # normalize observation and make it 1D
        nobs = self.normalizer.normalize_data(obs_seq, stats=self.norm_stats["obs"])
        nobs = nobs.flatten()
        # concat obstacle information to observations
        nobs = np.concatenate([nobs] + obs_dict["obs_encode"], axis=0)
        # device transfer
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (1, obs_horizon*obs_dim+obstacle_encode_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)

            # denoise
            denoise_timesteps = (
                self.noise_scheduler.timesteps[:1] if self.use_single_step_inference else self.noise_scheduler.timesteps
            )
            for k in denoise_timesteps:
                # predict noise
                noise_pred = self.net(sample=naction, timestep=k, global_cond=obs_cond)
                # inverse diffusion step (remove noise)
                if self.use_single_step_inference:
                    naction = noisy_action - noise_pred
                else:
                    results = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction)
                    naction = results[0] # (1, pred_horizon, action_dim)
                    mu_t = results[2]
                    # TODO: how to pass mu instead of naction (mu+var)

                if self.use_ee2d_guidance:
                    diffusing_action = self.normalizer.unnormalize_data(
                        naction.detach().to("cpu").numpy().squeeze(), stats=self.norm_stats["act"]
                    )  # (pred_horizon, 2)
                    mu_t = self.normalizer.unnormalize_data(
                        mu_t.detach().to("cpu").numpy().squeeze(), stats=self.norm_stats["act"]
                    )
                    if k < self.ee2d_guidance.denoising_guidance_step:
                        refined_action = diffusing_action.copy()
                        mu_action = mu_t.copy()

                        # for idx, act in enumerate(diffusing_action):
                        obstacle_info, pred_action = self._preprocess_guidance_input(
                            obs_dict, mu_action
                        )
                        cost_gradients = self.ee2d_guidance.compute_cost_gradients(
                            obstacle_info=obstacle_info,
                            pred_action=pred_action,
                        )
                        refined_action += cost_gradients
                        # refined_action[idx, ...] = self._calculate_refined_action_step(act, safe_yz_velocity)
                        # TODO: colculate refined action here

                        naction = self.normalizer.normalize_data(np.array(refined_action), stats=self.norm_stats["act"])
                        naction = torch.from_numpy(naction).to(self.device, dtype=torch.float32).unsqueeze(0)

        # unnormalize action
        naction = naction.detach().to("cpu").numpy()
        # (1, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = self.normalizer.unnormalize_data(naction, stats=self.norm_stats["act"])

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end, :]  # (action_horizon, action_dim)

        return action

    def load_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def set_config(self, config: Dict):
        self.obs_horizon = config["obs_horizon"]
        self.action_horizon = config["action_horizon"]
        self.pred_horizon = config["pred_horizon"]
        self.action_dim = config["controller"]["networks"]["action_dim"]
        self.sampling_time = config["controller"]["common"]["sampling_time"]
        self.norm_stats = {
            "act": config["normalizer"]["action"],
            "obs": config["normalizer"]["observation"],
        }
        self.quadrotor_params = config["simulator"]
        self.use_single_step_inference = config.get("controller").get("common").get("use_single_step_inference", False)

    def calculate_force_command(self, state: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
        y, y_dot, z, z_dot, phi, phi_dot = state
        yr, yr_dot, zr, zr_dot, phir, phir_dot = ref_state
        (
            dt,
            m_q,
        ) = self.quadrotor_params["dt"], self.quadrotor_params["m_q"]
        g, I_xx = self.quadrotor_params["g"], self.quadrotor_params["I_xx"]
        # how on earth do you want to calculate acceleration from position signals
        # est_zr_dot = (zr - z) / dt
        # est_phir_dot = (phir - phi) / dt
        zr_ddot = (zr_dot - z_dot) / dt
        phir_ddot = (phir_dot - phi_dot) / dt
        return np.array([m_q * (g + zr_ddot), I_xx * phir_ddot])

    def _preprocess_guidance_input(
        self, obs_dict: Dict[str, List], diffusing_action: np.ndarray
    ):
        """
        Return:
            obstacle_info: {"center": list, "radius": list}
            diffusing_action: (pred_horizon, 2)
        """
        obstacles = obs_dict["obs_encode"][0].reshape(3, 4)
        centers = obstacles[:, :2].tolist()
        radii = obstacles[:, 2:].tolist()
        obstacle_info = {"center": centers, "radius": radii}
        return obstacle_info, diffusing_action

    def _calculate_refined_action_step(self, pred_act, safe_yz_velocity):
        refined_step_action = pred_act.copy()
        refined_step_action[0] += safe_yz_velocity[0] * self.sampling_time
        refined_step_action[2] += safe_yz_velocity[1] * self.sampling_time
        refined_step_action[1] = safe_yz_velocity[0]
        refined_step_action[3] = safe_yz_velocity[1]
        refined_step_action[4] = -np.arctan(safe_yz_velocity[0] / safe_yz_velocity[1])
        # refinedstep_action[5] = (refinedstep_action[4] - pred_act[4]) / self.sampling_time
        return refined_step_action
