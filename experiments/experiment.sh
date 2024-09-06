#!/bin/bash

#SBATCH --job-name=one_policy_to_run_them_all
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=49
#SBATCH --mem-per-cpu=2000
#SBATCH --time=2-23:59:59

conda activate one_policy_to_run_them_all

python experiment.py \
    --algorithm.name="uni_ppo.ppo" \
    --algorithm.total_timesteps=1600143360 \
    --algorithm.nr_steps=10880 \
    --algorithm.minibatch_size=32640 \
    --algorithm.nr_epochs=10 \
    --algorithm.start_learning_rate=0.0004 \
    --algorithm.end_learning_rate=0.0 \
    --algorithm.entropy_coef=0.0 \
    --algorithm.gae_lambda=0.9 \
    --algorithm.critic_coef=1.0 \
    --algorithm.max_grad_norm=5.0 \
    --algorithm.clip_range=0.1 \
    --algorithm.evaluation_episodes=50 \
    --algorithm.evaluation_frequency=17233920 \
    --algorithm.save_latest_frequency=17233920 \
    --algorithm.determine_fastest_cpu_for_gpu=True \
    --algorithm.device="gpu" \
    --environment.name="multi_robot" \
    --environment.nr_envs=48 \
    --environment.async_skip_percentage=0.0 \
    --environment.cycle_cpu_affinity=True \
    --environment.seed=0 \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="placeholder" \
    --runner.project_name="one_policy_to_run_them_all" \
    --runner.exp_name="E0" \
    --runner.notes=""
