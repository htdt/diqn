import torch
import numpy as np

from core.buffer import ReplayBuffer
from core.make_env import make_vec_envs
from core.conv import Conv
from core.optim import LRParam
from iqn.algo import IQNAgent
from iqn.model import IQN


def get_buffer(cfg, env, device):
    return ReplayBuffer(
        size=cfg['buffer']['size'],
        batch_size=cfg['buffer']['batch_size'],
        obs_shape=env.observation_space.shape,
        device=device,
        obs_dtype=(torch.uint8 if 'conv' in cfg else torch.float),
        num_env=cfg['env']['num'],
        gamma=float(cfg['agent']['gamma']),
        n_step=cfg['agent']['n_step'],
    )


def get_envs(cfg, device):
    return make_vec_envs(
        env_name=cfg['env']['name'],
        num_processes=cfg['env']['num'],
        device=device,
        num_frame_stack=cfg['env'].get('frame_stack', 1),
    )


def get_agent(cfg, env, device):
    conv = None if 'conv' not in cfg else\
        Conv(**cfg['conv'], input_size=env.observation_space.shape)

    qf = IQN(
        conv=conv,
        num_samples=cfg['agent']['quantile_samples'],
        num_samples_eval=cfg['agent']['quantile_samples_eval'],
        hidden_sizes=cfg['agent']['hidden_sizes'],
        quantile_dim=cfg['agent']['quantile_dim'],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
        device=device,
    ).to(device=device).train()

    return IQNAgent(qf=qf, lr=LRParam(**cfg['lr']),
                    target_tau=cfg['agent']['target_tau'])
