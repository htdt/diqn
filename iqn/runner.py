import random
from dataclasses import dataclass
import torch
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from iqn.model import IQN
from core.exploration import DecayingEpsilon


@dataclass
class EnvRunner:
    envs: ShmemVecEnv
    qf: IQN
    eps: DecayingEpsilon
    device: str

    def _act(self, obs):
        with torch.no_grad():
            a = self.qf.argmax_actions(obs)
        for i in range(self.envs.num_envs):
            if random.random() < self.eps():
                a[i] = self.envs.action_space.sample()
        return a

    def __iter__(self):
        ep_reward, ep_len = [], []
        obs = self.envs.reset()
        yield {'obs': obs}

        while True:
            action = self._act(obs).unsqueeze(1)
            obs, reward, terminal, infos = self.envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    ep_reward.append(info['episode']['r'])
                    ep_len.append(info['episode']['l'])

            if len(ep_reward) >= self.envs.num_envs:
                ep_info = {
                    'episode/reward': sum(ep_reward) / len(ep_reward),
                    'episode/len': sum(ep_len) / len(ep_len),
                }
                ep_reward.clear(), ep_len.clear()
            else:
                ep_info = None

            yield {'obs': obs, 'action': action, 'reward': reward,
                   'terminal': terminal, 'ep_info': ep_info}
