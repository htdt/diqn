import sys
import yaml
import time
import moviepy.editor as mpy
from from_config import make_vec_envs, get_agent
from common.find_checkpoint import find_checkpoint


def render(cfg_name, steps, to_gif=False):
    with open(f'config/{cfg_name}.yaml') as f:
        cfg = yaml.load(f)

    env = make_vec_envs(env_name=cfg['env']['name'], num_processes=1,
                        num_frame_stack=cfg['env'].get('frame_stack', 1))
    agent = get_agent(cfg, env, 'cpu')
    _, fname = find_checkpoint(cfg)
    assert fname is not None
    agent.load(fname, map_location='cpu')
    agent.qf.eval()
    print(f'running {fname}')

    obs = env.reset()
    obs_stack = []
    for _ in range(steps):
        if to_gif:
            obs_stack.append(obs[0, 0, :, :, None].clone().numpy())
        else:
            env.render()
            time.sleep(1/30)

        action = agent.qf.argmax_actions(obs).unsqueeze(1)
        obs = env.step(action)[0]

    if to_gif:
        clip = mpy.ImageSequenceClip(obs_stack, fps=20)
        clip.write_gif(f'{cfg_name}.gif', verbose=True)


if __name__ == '__main__':
    assert len(sys.argv) in [3, 4], 'config name and steps number required'
    to_gif = len(sys.argv) == 4 and sys.argv[3] == 'gif'
    render(sys.argv[1], int(sys.argv[2]), to_gif)
