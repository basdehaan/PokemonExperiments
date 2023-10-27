from os.path import exists
from pathlib import Path
import uuid
from gold_gym_env import GoldGymEnv
from stable_baselines3 import A2C, PPO
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

    def _init():
        env = GoldGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    ep_length = 2 ** 11
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    print(sess_path)
    # init state: '../has_pokedex_nballs.state'

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': None, 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonGold.gbc', 'debug': False, 'sim_frame_dist': 1_000_000.0,
        'use_screen_explore': True, 'extra_buttons': False, 'explore_weight': 3
    }

    num_cpu = 24  # 64 #46  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                             name_prefix='poke')
    # env_checker.check_env(env)
    learn_steps = 40
    file_name = './session_518b06be/poke_221184_steps.zip'
    file_name = file_name.replace(".zip", "")
    if exists(file_name + '.zip'):
        print('loading checkpoint', file_name)
        print()
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print('invalid checkpoint', file_name)
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999,
                    learning_rate=0.01)

    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length * num_cpu * 1000, callback=checkpoint_callback)
