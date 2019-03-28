import sys
import torch
import yaml
from tqdm import trange
from core.exploration import DecayingEpsilon
from common.find_checkpoint import find_checkpoint
from common.logger import Logger
from iqn.runner import EnvRunner
from from_config import get_agent, get_buffer, get_envs


def train(cfg_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'running on {device}')
    with open(f'config/{cfg_name}.yaml') as f:
        cfg = yaml.load(f)

    if cfg['train'].get('resume', False):
        n_start, fname = find_checkpoint(cfg)
    else:
        n_start, fname = 0, None

    eps = DecayingEpsilon(**cfg['exploration'], n_iter=n_start)
    log = Logger(eps=eps, device=device)
    envs = get_envs(cfg, device)
    buffer = get_buffer(cfg, envs, device)
    agent = get_agent(cfg, envs, device)
    if fname:
        agent.load(fname)
        agent.init_target()
    env_runner = EnvRunner(envs=envs, qf=agent.qf, device=device, eps=eps)

    cp_iter = cfg['train']['checkpoint_every']
    log_iter = cfg['train']['log_every']
    counter = trange(n_start, cfg['train']['steps'] + 1)

    for n_iter, env_step in zip(counter, env_runner):
        eps.update(n_iter)
        buffer.append(env_step)
        if env_step.get('ep_info') is not None:
            log.output(env_step['ep_info'], n_iter)

        lr = agent.optim.step_lr(n_iter)
        if lr:
            log.output({'lr': lr}, n_iter)

        if len(buffer) > cfg['buffer']['warmup']:
            to_log = agent.update(buffer.sample())
            if n_iter % log_iter == 0:
                log.stats(n_iter)
                log.output(to_log, n_iter)

        if n_iter > n_start and n_iter % cp_iter == 0:
            fname = cfg['train']['checkpoint_name'].format(
                n_iter=n_iter//cp_iter)
            agent.save(fname)


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'config name required'
    train(sys.argv[1])
