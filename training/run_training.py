import os
import random
from os.path import exists
from pathlib import Path
# import uuid

from gold_env import GoldGymEnv
from stable_baselines3 import PPO, A2C
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
    env_conf_copy = env_conf.copy()
    env_conf_copy['seed'] = seed
    assert env_conf_copy['class'] is not None

    def _init():
        _env = env_conf_copy['class'](env_conf_copy)
        _env.reset(seed=(seed + i))
        return _env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    ep_length = 100
    reset_length = 100 * ep_length
    num_emulators = 8
    visible_emulators = 2
    episodes = 10_000

    learning_rate_min = 0.0001
    learning_rate_max = 0.002
    n_epochs = 1
    batch_size = 64

    types = [
        GoldGymEnv,
        RedGymEnv,
    ]

    sess_path = Path("../_session_continuous")
    print(sess_path)

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 80, 'load_once': True, 'random_reload': 0, 'rolling_reload': int(reset_length / ep_length),
        'max_steps': ep_length,
        'save_stats_and_runs': False, 'random_init_state': True,
        'print_rewards': False, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'debug': False, 'sim_frame_dist': 500_000.0,
        'explore_method': 'HYBRID', 'extra_buttons': False, 'explore_weight': 1,
        'noise': 0.0
    }


    def get_env_config_for_i(i, rand=False):
        _env = env_config.copy()
        if rand:
            _env['class'] = random.choice(types)
        else:
            _env['class'] = types[i % len(types)]
        # if i < len(set(types)) or i < visible_emulators:
        if i < visible_emulators:
            # visible windows
            _env['headless'] = False
            _env['random_reload'] = 0
            # _env['rolling_reload'] = -1
            _env['class'] = types[i % len(types)]
        _env['class_indicator'] = types.index(_env['class']) / len(set(types))
        return _env


    env = SubprocVecEnv([make_env(i, get_env_config_for_i(i), seed=1234) for i in
                         range(num_emulators)], start_method="spawn")

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
        # net_arch=[1024, 256, 128, 64]
    )
    gamma = 0.9926

    # agent = PPO('CnnPolicy', env, n_steps=ep_length, batch_size=batch_size, n_epochs=n_epochs,learning_rate=learning_rate, policy_kwargs=policy_kwargs, gamma=gamma)
    agent = A2C('CnnPolicy', env, n_steps=ep_length, policy_kwargs=policy_kwargs, gamma=gamma)

    if len(files) > 0:
        file_name = f'{search_folder}/{files[0]}'
        file_name = file_name.replace(".zip", "")
        if exists(file_name + '.zip'):
            print('loading checkpoint', file_name)
            print()
            if type(agent) is PPO:
                agent = PPO.load(file_name, env=env, gamma=gamma)
                agent.batch_size = batch_size
                agent.n_steps = ep_length
                agent.n_epochs = n_epochs
                agent.learning_rate = learning_rate_min + (random.random() * (learning_rate_max - learning_rate_min))
                agent.rollout_buffer.buffer_size = ep_length
                agent.rollout_buffer.n_envs = num_emulators
                agent.rollout_buffer.reset()
            elif type(agent) is A2C:
                agent = A2C.load(file_name, env=env, gamma=gamma)
                agent.n_steps = ep_length
                agent.n_epochs = n_epochs
                agent.rollout_buffer.buffer_size = ep_length
                agent.rollout_buffer.n_envs = num_emulators
                agent.rollout_buffer.reset()

    for i in range(episodes):
        print(i + 1, "/", episodes)
        agent.learn(total_timesteps=ep_length * num_emulators,
                    callback=checkpoint_callback,
                    reset_num_timesteps=False,
                    progress_bar=True,
                    log_interval=num_emulators)
