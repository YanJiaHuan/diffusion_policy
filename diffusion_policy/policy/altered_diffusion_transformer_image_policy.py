# created by jh
from typing import Dict, Tuple
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class DiffusionTransformerImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            loss_fn,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            n_layer=8,
            n_head=4,
            n_emb=256,
            # n_emb=768,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            **kwargs):
        super().__init__()
        self.n_emb = n_emb
        self.loss = loss_fn
        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=self.n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs
        self.num_train_timesteps = noise_scheduler.config.num_train_timesteps

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
         # Print parameter counts
        # self.print_model_parameters()
        self.reset_mse_per_timestep()

    def reset_mse_per_timestep(self):
        self.mse_per_timestep = {t: [] for t in
                                 range(self.num_train_timesteps)}

    # ========= inference  ============
    #+-----------------------------------------------+
    def print_model_parameters(self):
            obs_encoder_params = sum(p.numel() for p in self.obs_encoder.parameters())
            model_params = sum(p.numel() for p in self.model.parameters())
            print(f'obs_encoder parameter count: {obs_encoder_params}')
            print(f'diffusion model parameter count: {model_params}')
    #+-----------------------------------------------+
    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None, generator=None,
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                # **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs)

        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            obs_encoder_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # This converts the two-sampled episodes into the shape
            # [n_obs_steps * B, C, B, H]
            this_nobs = dict_apply(nobs,
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # This then converts the feature outputs back to include a dimension
            # for n_obs_steps
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # This generates a mask of the same dimensions, stating True if the
        # action is within the max obs steps range, and is actually an action
        # observation, otherwise it will be False
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # This is then creating noise along the action/trajectory space
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # This is then stating the timesteps for noise scheduling, but it is
        # setting random timesteps for each batch
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # loss = F.mse_loss(pred, target, reduction='none')
        loss = self.loss(pred, target)
        loss = loss * loss_mask.type(loss.dtype)
        for t, l in zip(timesteps, loss):
            t_value = t.detach().cpu().item()  # Ensure this is a scalar
            mean_l_value = l.mean().detach().cpu().item()
            self.mse_per_timestep[t_value].append(mean_l_value)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss