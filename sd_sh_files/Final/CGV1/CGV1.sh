#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH --ntasks-per-node=1

#SBATCH --gpus=1

#SBATCH -t 2-00:00:00

#SBATCH --cpus-per-task=30

#SBATCH -o /users/%u/SafetyAwareModelBasedRL/logs/%j.out

#SBATCH --mem=500G

source ~/.bashrc

module load gcc/10.3.0-gcc-9.4.0

module load mesa/22.0.2-gcc-10.3.0-python3+-chk-version

module load llvm/12.0.1-gcc-10.3.0-python3+-chk-version

module load cuda/11.7.0-gcc-10.3.0

module load cudnn/8.2.4.15-11.4-gcc-10.3.0

eval "$(conda shell.bash hook)"

conda activate safetyGym

CUDA_VISIBLE_DEVICES=0 
python ~/SafetyAwareModelBasedRL/dreamer.py --configs defaults --logdir /scratch/users/k21176641/final/CG1/ar_3 --action_repeat 3 --imag_horizon 20 --lag_clip 10 --steps 2e6 --lag_grad_clip 1000 --actor_entropy 6e-4 --task Safexp-CarGoal1-v0 --dyn_hidden 2048 --dyn_deter 2048 --units 768 --reward_layers 3 --cost_layers 3 --discount_layers 3 --value_layers 4 --actor_layers 5 --cnn_depth 96



