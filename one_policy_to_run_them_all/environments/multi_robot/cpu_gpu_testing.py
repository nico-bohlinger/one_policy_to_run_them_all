import time
import subprocess
import psutil
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
from flax.training.train_state import TrainState


NR_STEPS = 1000
DIM_1 = 45
DIM_2 = 1024


class NeuralNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        policy_mean = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        policy_mean = nn.tanh(policy_mean)
        policy_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.tanh(policy_mean)
        policy_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.tanh(policy_mean)
        policy_mean = nn.Dense(18, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_logstd = self.param("policy_logstd", constant(jnp.log(1.0)), (1, 18))
        return policy_mean, policy_logstd


def determine_fastest_cpu_for_gpu_connection(cpus):
    times = {}

    for cpu_id in cpus:
        if cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([cpu_id,])

        key = jax.random.PRNGKey(0)

        key, subkey = jax.random.split(key)

        nn = NeuralNetwork()
        nn.apply = jax.jit(nn.apply)

        x = np.random.rand(DIM_1, DIM_2)

        nn_state = TrainState.create(apply_fn=nn.apply, params=nn.init(subkey, x), tx=optax.adam(1e-3))

        @jax.jit
        def action_selection(nn_state, x, key):
            action_mean, action_logstd = nn.apply(nn_state.params, x)
            action_std = jnp.exp(action_logstd)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            return action, key

        for i in range(NR_STEPS + 1):
            if i == 1:
                start_time = time.time()

            x = np.random.rand(DIM_1, DIM_2)
            action, key = action_selection(nn_state, x, key)
            something = x + np.sum(action)

        end_time = time.time()
        
        times[cpu_id] = end_time - start_time
    
    fastest_cpu = min(times, key=times.get)

    del times
    del x
    del action
    del something
    del nn_state
    del nn
    del key
    del subkey

    psutil.Process().cpu_affinity([])

    return fastest_cpu


def get_global_cpu_ids():
    # Execute the lstopo command to get CPU details
    lstopo_output = subprocess.check_output(["lstopo", "--only", "pu"], text=True)

    # Parse the output to extract global CPU IDs
    global_cpu_ids = []
    for line in lstopo_output.split('\n'):
        if line.startswith("PU L#"):
            # Extracting the global CPU ID (P#) from the line
            p_index = line.find("(P#")
            if p_index != -1:
                # Extracting and converting the CPU ID to integer
                cpu_id = int(line[p_index+3:line.find(")", p_index)])
                global_cpu_ids.append(cpu_id)
    
    return global_cpu_ids


def get_fastest_cpu_for_gpu_connection(global_cpu_ids):
    fastest_cpu_id = determine_fastest_cpu_for_gpu_connection(global_cpu_ids)

    return fastest_cpu_id
