import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(config.algorithm.softmax_temperature, config.algorithm.softmax_temperature_min, config.algorithm.stability_epsilon)


class Critic(nn.Module):
    softmax_temperature: float
    softmax_temperature_min: float
    stability_epsilon: float

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

        value = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(combined_input)
        value = nn.LayerNorm()(value)
        value = nn.elu(value)
        value = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
        value = nn.elu(value)
        value = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
        value = nn.elu(value)
        value = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(value)
        return value
