import os
import random
from os.path import exists
from pathlib import Path
import numpy as np

from imitation.algorithms import bc
from imitation.data import types

from gold_env import GoldGymEnv
from red_env import RedGymEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

# import torch
# print("Using GPU:", torch.cuda.is_available())
# print("Using:", torch.cuda.get_device_name(torch.cuda.current_device()))

import pyboy
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

    ep_length = 200
    reset_length = 10 * ep_length
    num_emulators = 16
    visible_emulators = 2
    episodes = 1000

    learning_rate = 0.0005
    n_epochs = 2
    batch_size = 64

    human_emulator = True
    human_emulator_interval = 10
    observations = []
    actions = []

    classes = [
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
        'explore_method': 'STEPS', 'extra_buttons': False, 'explore_weight': 1,
        'noise': 0.0
    }


    def get_env_config_for_i(i=0, human=False):
        _env = env_config.copy()
        if human:
            _env['class'] = random.choice(classes)
        else:
            _env['class'] = classes[i % len(classes)]

        if i < visible_emulators:
            # visible windows
            _env['headless'] = False
            _env['random_reload'] = 0
            # _env['rolling_reload'] = -1

        if human:
            _env['headless'] = False
            _env['random_reload'] = 1

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
    agent = A2C('MultiInputPolicy', env, n_steps=ep_length, policy_kwargs=policy_kwargs, gamma=gamma,
                learning_rate=learning_rate)

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
                agent.learning_rate = learning_rate
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

    if human_emulator:
        def obs_convert(obs):
            return {
                "battle_type": int(obs["battle_type"]),
                "party_status": np.array(obs["party_status"], dtype=np.float64),
                "screen_image": np.array(np.transpose(obs["screen_image"], (2, 0, 1)), dtype=np.uint8),
                "version": int(obs["version"]),
            }


        env_settings = get_env_config_for_i(human=True)
        env_settings['random_reload'] = 1
        human_env = make_env(-1, env_settings, seed=0)()
        obs, info = human_env.reset()
        observations.append(obs_convert(obs))
        human_env.pyboy.set_emulation_speed(4)

    for i in range(episodes):
        print(i + 1, "/", episodes)
        agent.learn(total_timesteps=ep_length * num_emulators,
                    callback=checkpoint_callback,
                    reset_num_timesteps=False,
                    progress_bar=True,
                    log_interval=num_emulators)

        if human_emulator and i % human_emulator_interval == 0:
            for a in range(ep_length):
                action = []
                while len(action) == 0 or action[0] not in human_env.valid_actions:
                    human_env.pyboy.tick()
                    action = human_env.pyboy.get_input()

                action = human_env.valid_actions.index(action[0])
                actions.append(action)
                obs, reward, done, truncated, info = human_env.step(action)
                observations.append(obs_convert(obs))

            # print(agent.observation_space)
            # print("Observation structure:")
            # for k, v in observations[1].items():
            #     print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")

            observations = [types.DictObs(o) for o in observations]

            traj = types.Trajectory(obs=np.array(observations), acts=np.array(actions, dtype=np.int64), infos=None,
                                    terminal=True)
            bc_trainer = bc.BC(observation_space=agent.observation_space, action_space=agent.action_space,
                               demonstrations=[traj], policy=agent.policy, rng=np.random.default_rng())
            bc_trainer.train(n_epochs=100)

            human_env.reset()
            human_env.pyboy.tick()
