from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (
            Policy(
                config.algorithm.std_dev,
                config.algorithm.softmax_temperature, config.algorithm.softmax_temperature_min,
                config.algorithm.stability_epsilon,
                config.algorithm.policy_mean_abs_clip,
                config.algorithm.policy_std_min_clip, config.algorithm.policy_std_max_clip
            ), 
            get_processed_action_function()
        )


class Policy(nn.Module):
    std_dev: float
    softmax_temperature: float
    softmax_temperature_min: float
    stability_epsilon: float
    policy_mean_abs_clip: float
    policy_std_min_clip: float
    policy_std_max_clip: float

    @nn.compact
    def __call__(self, dynamic_joint_description, dynamic_joint_state, dynamic_foot_description, dynamic_foot_state, general_state):
        joint_log_softmax_temperature = self.param("joint_log_softmax_temperature", constant(jnp.log(self.softmax_temperature - self.softmax_temperature_min)), (1,))
        dynamic_joint_state_mask = nn.Dense(64)(dynamic_joint_description)
        dynamic_joint_state_mask = nn.LayerNorm()(dynamic_joint_state_mask)
        dynamic_joint_state_mask = nn.elu(dynamic_joint_state_mask)
        dynamic_joint_state_mask = nn.Dense(64)(dynamic_joint_state_mask)
        dynamic_joint_state_mask = jnp.clip(nn.tanh(dynamic_joint_state_mask), -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)
        latent_dynamic_joint_state = nn.Dense(4)(dynamic_joint_state)
        latent_dynamic_joint_state = nn.elu(latent_dynamic_joint_state)
        joint_e_x = jnp.exp(dynamic_joint_state_mask / (jnp.exp(joint_log_softmax_temperature) + self.softmax_temperature_min))
        dynamic_joint_state_mask = joint_e_x / (joint_e_x.sum(axis=-1, keepdims=True) + self.stability_epsilon)
        dynamic_joint_state_mask = jnp.repeat(jnp.expand_dims(dynamic_joint_state_mask, axis=-1), latent_dynamic_joint_state.shape[-1], axis=-1)
        masked_dynamic_joint_state = dynamic_joint_state_mask * jnp.expand_dims(latent_dynamic_joint_state, axis=-2)
        masked_dynamic_joint_state = jnp.reshape(masked_dynamic_joint_state, masked_dynamic_joint_state.shape[:-2] + (masked_dynamic_joint_state.shape[-2] * masked_dynamic_joint_state.shape[-1],))
        dynamic_joint_latent = jnp.sum(masked_dynamic_joint_state, axis=-2)

        foot_log_softmax_temperature = self.param("foot_log_softmax_temperature", constant(jnp.log(self.softmax_temperature - self.softmax_temperature_min)), (1,))
        dynamic_foot_state_mask = nn.Dense(32)(dynamic_foot_description)
        dynamic_foot_state_mask = nn.LayerNorm()(dynamic_foot_state_mask)
        dynamic_foot_state_mask = nn.elu(dynamic_foot_state_mask)
        dynamic_foot_state_mask = nn.Dense(32)(dynamic_foot_state_mask)
        dynamic_foot_state_mask = jnp.clip(nn.tanh(dynamic_foot_state_mask), -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)
        latent_dynamic_foot_state = nn.Dense(4)(dynamic_foot_state)
        latent_dynamic_foot_state = nn.elu(latent_dynamic_foot_state)
        foot_e_x = jnp.exp(dynamic_foot_state_mask / (jnp.exp(foot_log_softmax_temperature) + self.softmax_temperature_min))
        dynamic_foot_state_mask = foot_e_x / (foot_e_x.sum(axis=-1, keepdims=True) + self.stability_epsilon)
        dynamic_foot_state_mask = jnp.repeat(jnp.expand_dims(dynamic_foot_state_mask, axis=-1), latent_dynamic_foot_state.shape[-1], axis=-1)
        masked_dynamic_foot_state = dynamic_foot_state_mask * jnp.expand_dims(latent_dynamic_foot_state, axis=-2)
        masked_dynamic_foot_state = jnp.reshape(masked_dynamic_foot_state, masked_dynamic_foot_state.shape[:-2] + (masked_dynamic_foot_state.shape[-2] * masked_dynamic_foot_state.shape[-1],))
        dynamic_foot_latent = jnp.sum(masked_dynamic_foot_state, axis=-2)

        combined_input = jnp.concatenate([dynamic_joint_latent, dynamic_foot_latent, general_state], axis=-1)

        action_latent = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(combined_input)
        action_latent = nn.LayerNorm()(action_latent)
        action_latent = nn.elu(action_latent)
        action_latent = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(action_latent)
        action_latent = nn.elu(action_latent)
        action_latent = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(action_latent)

        action_description_latent = nn.Dense(128)(dynamic_joint_description)
        action_description_latent = nn.LayerNorm()(action_description_latent)
        action_description_latent = nn.elu(action_description_latent)
        action_description_latent = nn.Dense(128)(action_description_latent)

        action_latent = jnp.expand_dims(action_latent, axis=-2)
        action_latent = jnp.broadcast_to(action_latent, shape=(*action_latent.shape[:-2], action_description_latent.shape[-2], action_latent.shape[-1]))
        combined_action_latent = jnp.concatenate([action_latent, stop_gradient(latent_dynamic_joint_state), action_description_latent], axis=-1)
        policy_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(combined_action_latent)
        policy_mean = nn.LayerNorm()(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_mean = jnp.clip(policy_mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)

        policy_logstd = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(np.log(self.std_dev)))(action_description_latent)
        policy_logstd = jnp.clip(policy_logstd, np.log(self.policy_std_min_clip), np.log(self.policy_std_max_clip))

        return policy_mean.squeeze(-1), policy_logstd.squeeze(-1)


def get_processed_action_function():
    def get_clipped_and_scaled_action(action):
        return action
    return jax.jit(get_clipped_and_scaled_action)
