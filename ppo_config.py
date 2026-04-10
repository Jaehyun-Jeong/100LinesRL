import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless MuJoCo/DM Control rendering

import torch as T
import argparse

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

ENV_CONFIGS = {
    # --------------------------------------------------------------------------
    # 1. Gymnasium
    # --------------------------------------------------------------------------
    'CartPole-v1': {
        # [target score] 500.0
        'lr': 2.5e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.1,
        'max_grad_norm': 0.5,
        'K_epoch': 3,
        'T_horizon': 128,
        'mb_size': 32,
        'n_envs': 4,
        'max_timesteps': 100_000,
        'vf_coef': 0.5,
        'ent_coef': 0.01,
    },

    # --------------------------------------------------------------------------
    # 2. Gymnasium (Continuous)
    # --------------------------------------------------------------------------
    'LunarLanderContinuous-v3': {
        # [target score] >= 200.0
        'lr': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64,
        'n_envs': 4,
        'max_timesteps': 1_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.0,
    },

    # --------------------------------------------------------------------------
    # 3. MuJoCo (Continuous)
    # --------------------------------------------------------------------------
    'Hopper-v5': {
        # [target score] 2,500 ~ 3,000
        'lr': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64,
        'n_envs': 4,
        'max_timesteps': 1_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.0
    },

    'Walker2d-v5': {
        # [target score] 3,000 ~ 4,000
        'lr': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64,
        'n_envs': 4,
        'max_timesteps': 1_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.0
    },

    # --------------------------------------------------------------------------
    # 4. DeepMind Control (Continuous)
    # --------------------------------------------------------------------------
    'dm_control/cheetah-run-v0': {
        # [target score] 700 ~ 800
        'lr': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64,
        'n_envs': 4,
        'max_timesteps': 5_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.01
    },

    'dm_control/walker-walk-v0': {
        # [target score] >= 900
        'lr': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64,
        'n_envs': 4,
        'max_timesteps': 2_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.01
    }
}

# ==============================================================================
# User Selection
# ==============================================================================

parser = argparse.ArgumentParser(description='PPO Hyperparameters')
parser.add_argument('--env', type=str, default="dm_control/cheetah-run-v0", help='Environment name')
args = parser.parse_args()
env_name = args.env
'''CartPole-v1, LunarLanderContinuous-v3, Hopper-v5, Walker2d-v5, dm_control/cheetah-run-v0, dm_control/walker-walk-v0'''

default_cfg = ENV_CONFIGS['LunarLanderContinuous-v3']

if env_name in ENV_CONFIGS:
    cfg = ENV_CONFIGS[env_name]
else:
    print(f"[Warning] '{env_name}' Use default environment(Pendulum-v1).")
    cfg = default_cfg

lr = cfg['lr']
gamma = cfg['gamma']
lmbda = cfg['lmbda']
eps_clip = cfg['eps_clip']
K_epoch = cfg['K_epoch']
T_horizon = cfg['T_horizon']
mb_size = cfg['mb_size']
max_grad_norm = cfg['max_grad_norm']
max_timesteps = cfg['max_timesteps']
vf_coef = cfg['vf_coef']
ent_coef = cfg['ent_coef']
n_envs = cfg['n_envs']
