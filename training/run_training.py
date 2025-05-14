import os
import random
from os.path import exists
from pathlib import Path
import uuid

from gold_env import GoldGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

import torch

print("Using GPU:", torch.cuda.is_available())
print("Using:", torch.cuda.get_device_name(torch.cuda.current_device()))

import pyboy

pyboy.logger.log_level("DISABLE")


def make_env(i, env_conf, seed=0):
    """
    Utility function for parallel envs.
    :param i: index of the subprocess
    :param env_conf: settings for the env
    :param seed: (int) the initial seed for RNG
    """
    if i != 0 or seed == 0:
        seed = random.randint(0, 10000)

    def _init():
        _env = GoldGymEnv(env_conf)
        _env.reset(seed=(seed + i))
        return _env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    ep_length = 100
    num_cpu = 32
    episodes = 1000
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    # init_state = '../PokemonGold_chose_totodile_nickname_done.gbc.state'
    try:
        sess_path = Path("../_session_continuous")
        save_states = os.listdir('../_session_continuous/final_states')
        save_state_scores = [s[1:s.index("_")] for s in save_states]
        save_state_scores = [float(s) for s in save_state_scores]
        index = save_state_scores.index(max(save_state_scores))
        # init_state = '../training/_session_continuous/final_states/' + save_states[index]
    except Exception as e:
        print("exception", e)
        pass
    # print("loading init state", init_state)
    print(sess_path)

    env_config = {
        'headless': True, 'save_final_state': False, 'early_stop': False,
        'action_freq': 48, 'load_once': True, 'random_reload': 0, 'rolling_reload': -1,
        # 'init_state': init_state,
        'max_steps': ep_length,
        'save_stats_and_runs': False,
        'print_rewards': False, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonGold.gbc', 'debug': False, 'sim_frame_dist': 3_00_000.0,
        'explore_method': 'STEPS', 'extra_buttons': False, 'explore_weight': 1
    }

    gb_paths = ['../PokemonGold.gbc', '../PokemonRed.gb']
    
    def get_env_config_for_i(i):
        _env = env_config.copy()
        _env['gb_path'] = random.choice(gb_paths)
        if i < len(gb_paths):
            # n visible windows
            _env['headless'] = False
            _env['random_reload'] = 0
            _env['rolling_reload'] = -1
            _env['gb_path'] = gb_paths[i % len(gb_paths)]
        # _env['init_state'] = f'{gb_paths[i % len(gb_paths)]}.state'
        return _env


    env = SubprocVecEnv([make_env(i, get_env_config_for_i(i), seed=123) for i in
                         range(num_cpu)], start_method="spawn")

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=str(sess_path),
                                             name_prefix='poke')

    search_folder = "../_session_continuous"

    files = [f for f in os.listdir(search_folder) if 'poke' in f]

    files = sorted(files,
                   key=lambda x: int(str(x).replace('poke_', '').replace('_steps.zip', '')),
                   reverse=True)
    print(files)
    learning_rate = 0.05
    n_epochs = 30
    batch_size = 64

    # policy model shape
    policy_kwargs = dict(
        net_arch=[dict(pi=[1024, 1024, 256, 128, 64], vf=[1024, 1024, 256, 128, 64])]
    )
    agent = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=batch_size, n_epochs=n_epochs,
                learning_rate=learning_rate)
    if len(files) > 0:
        file_name = f'{search_folder}/{files[0]}'
        file_name = file_name.replace(".zip", "")
        if exists(file_name + '.zip'):
            print('loading checkpoint', file_name)
            print()
            agent = PPO.load(file_name, env=env)
            agent.batch_size = batch_size
            agent.n_steps = ep_length
            agent.n_epochs = n_epochs
            agent.n_envs = num_cpu
            agent.learning_rate = learning_rate
            agent.rollout_buffer.buffer_size = ep_length
            agent.rollout_buffer.n_envs = num_cpu
            agent.rollout_buffer.reset()

    for i in range(episodes):
        agent.learn(total_timesteps=ep_length * num_cpu,
                    callback=checkpoint_callback,
                    reset_num_timesteps=False,
                    progress_bar=True,
                    log_interval=num_cpu)
