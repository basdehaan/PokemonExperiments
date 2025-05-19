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

# import torch
# print("Using GPU:", torch.cuda.is_available())
# print("Using:", torch.cuda.get_device_name(torch.cuda.current_device()))

import pyboy

from training.red_env import RedGymEnv

pyboy.logger.log_level("DISABLE")


def make_env(i, env_conf, seed=0):
    if i != 0 or seed == 0:
        seed = random.randint(0, 10000)

    assert env_conf['class'] is not None

    def _init():
        _env = env_conf['class'](env_conf)
        _env.reset(seed=(seed + i))
        return _env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    ep_length = 500
    reset_length = 500
    num_cpu = 16
    episodes = 200

    learning_rate = 0.01
    n_epochs = 20
    batch_size = 64

    types = [
        GoldGymEnv,
        RedGymEnv
    ]

    sess_path = Path("../_session_continuous")
    print(sess_path)

    env_config = {
        'headless': True, 'save_final_state': False, 'early_stop': False,
        'action_freq': 100, 'load_once': True, 'random_reload': 0, 'rolling_reload': int(reset_length/ep_length),
        'max_steps': ep_length,
        'save_stats_and_runs': False,
        'print_rewards': False, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'debug': False, 'sim_frame_dist': 3_00_000.0,
        'explore_method': 'STEPS', 'extra_buttons': False, 'explore_weight': 1,
        'noise': 0.05
    }


    def get_env_config_for_i(i, rand=False):
        _env = env_config.copy()
        if rand:
            _env['class'] = random.choice(types)
        else:
            _env['class'] = types[i % len(types)]
        if i < len(types):
            # n visible windows
            _env['headless'] = False
            _env['random_reload'] = 0
            # _env['rolling_reload'] = -1
            _env['class'] = types[i % len(types)]
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

    # policy model shape
    policy_kwargs = dict(
        net_arch=[dict(pi=[1024, 256, 128, 64], vf=[1024, 256, 128, 64])]
    )
    agent = PPO('CnnPolicy', env, n_steps=ep_length, batch_size=batch_size, n_epochs=n_epochs,
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
