import os
import random
from os.path import exists
from pathlib import Path
import uuid
from typing import Callable

from gold_gym_env import GoldGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    if rank != 0 or seed == 0:
        seed = random.randint(0, 10000)

    def _init():
        env = GoldGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    ep_length = 5000
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    # sess_path = Path("_session_continuous")
    init_state = '../PokemonGold_chose_totodile.gbc.state'
    try:
        save_states = os.listdir('../baselines/_session_continuous/final_states')
        save_state_scores = [s[1:s.index("_")] for s in save_states]
        save_state_scores = [float(s) for s in save_state_scores]
        index = save_state_scores.index(max(save_state_scores))
        init_state = '../baselines/_session_continuous/final_states/' + save_states[index]
    except:
        pass
    print("loading init state", init_state)
    # TODO: find best saved state and use that
    print(sess_path)

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'load_once': True,
        'init_state': init_state,
        'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonGold.gbc', 'debug': False, 'sim_frame_dist': 2_00_000.0,
        'use_screen_explore': False, 'extra_buttons': False, 'explore_weight': 1
    }
    env_config_1 = env_config.copy()
    env_config_1['headless'] = False

    num_cpu = 1  # 64 #46  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config_1 if i % 12 == 0 else env_config, seed=672) for i in
                         range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=str(sess_path),
                                             name_prefix='poke')

    learn_steps = 25
    search_folder = "new_session"

    files = [f for f in os.listdir(f'../baselines/{search_folder}') if 'poke' in f]

    files = sorted(files,
                   key=lambda x: int(str(x).replace('poke_', '').replace('_steps.zip', '')),
                   reverse=True)
    print(files)
    agent = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=64, n_epochs=10,
                gamma=0.99,
                learning_rate=0.01)
    if len(files) > 0:
        file_name = f'{search_folder}/{files[0]}'
        file_name = file_name.replace(".zip", "")
        if exists(file_name + '.zip'):
            print('loading checkpoint', file_name)
            print()
            agent = PPO.load(file_name, env=env)
            agent.batch_size = 64
            agent.n_steps = ep_length
            agent.n_epochs = 30
            agent.n_envs = num_cpu
            agent.learning_rate = min(0.1, 0.003 * num_cpu)
            agent.rollout_buffer.buffer_size = ep_length
            agent.rollout_buffer.n_envs = num_cpu
            agent.rollout_buffer.reset()

    for i in range(learn_steps):
        agent.learn(total_timesteps=ep_length * num_cpu,
                    callback=checkpoint_callback,
                    reset_num_timesteps=False)
