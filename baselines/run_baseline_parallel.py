import os
import random
from os.path import exists
from pathlib import Path
import uuid
from typing import Callable

from gold_gym_env import GoldGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


def linear_schedule(minimum_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param minimum_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        lr = max(progress_remaining * minimum_value * 10, minimum_value)
        print(lr)
        return lr

    return func


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    if rank != 0:
        seed = random.randint(0, 10000)

    def _init():
        env = GoldGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    ep_length = 2 ** 12
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    print(sess_path)

    env_config = {
        'headless': True, 'save_final_state': False, 'early_stop': False,
        'action_freq': 24, 'init_state': '../PokemonGold_chose_totodile.gbc.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonGold.gbc', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': False, 'extra_buttons': True, 'explore_weight': 3
    }
    env_config_1 = env_config.copy()
    env_config_1['headless'] = False

    num_cpu = 12  # 64 #46  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config_1 if i < 1 else env_config, seed=8618) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=str(sess_path), name_prefix='poke')

    learn_steps = 100
    files = [f for f in os.listdir(f'../baselines/_session_start') if 'poke' in f]
    files = sorted(files, key=lambda x: int(str(x).replace('poke_', '').replace('_steps.zip','')), reverse=True)
    print(files)
    file_name = f'_session_start/{files[0]}'
    file_name = file_name.replace(".zip", "")
    if exists(file_name + '.zip'):
        print('loading checkpoint', file_name)
        print()
        model = PPO.load(file_name, env=env)
        model.batch_size = 64
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print('invalid checkpoint', file_name)
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=64, n_epochs=1, gamma=0.999,
                    learning_rate=linear_schedule(3e-4))

    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length * num_cpu, callback=checkpoint_callback, reset_num_timesteps=False)
