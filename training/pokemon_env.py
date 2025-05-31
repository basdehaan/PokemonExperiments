import random
import re
import sys
import uuid
from math import floor
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib
import mediapy as media

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent


class PokeGymEnv(Env):

    def __init__(
            self, config=None):

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320  # 1000
        self.headless = config['headless']
        self.num_elements = 20000  # max
        self.load_once = False if 'load_once' not in config else config['load_once']
        self.random_reload = 0 if 'random_reload' not in config else config['random_reload']
        self.rolling_reload = -1 if 'rolling_reload' not in config else config['rolling_reload']
        self.reload_roll = random.randint(0, self.rolling_reload - 1) if self.rolling_reload > 0 else 0
        self.loaded = False
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.explore_method = 'HYBRID' if 'explore_method' not in config else config['explore_method']
        self.noise = 0 if 'noise' not in config else config['noise']
        self.similar_frame_dist = config['sim_frame_dist']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PASS
        ]

        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PRESS_BUTTON_SELECT
            ])

        self.release_button = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        pixel_factor = 0.5
        self.output_shape = (int(144 * pixel_factor), int(160 * pixel_factor), 3)
        self.mem_padding = 0
        self.memory_height = 0
        self.col_steps = 16

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
            self.gb_path,
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(30)

        self.init_state = self.gb_path + ".state"
        init_state_file = Path(self.init_state)
        if not init_state_file.exists():
            self.init_state = None

        self.reset()

    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        if self.init_state:
            if not self.loaded:
                # initial load
                with open(self.init_state, "rb") as f:
                    self.pyboy.load_state(f)
                    self.init_knn_map()
            elif self.loaded and not self.load_once:
                # reload every time
                with open(self.init_state, "rb") as f:
                    self.pyboy.load_state(f)
                    self.init_knn_map()
            elif self.reload_roll == self.rolling_reload:
                # reload every x iterations
                with open(self.init_state, "rb") as f:
                    self.pyboy.load_state(f)
                    self.init_knn_map()
                self.reload_roll = 0
            elif random.random() < self.random_reload:
                # reload randomly
                with open(self.init_state, "rb") as f:
                    self.pyboy.load_state(f)
                    self.init_knn_map()
        else:
            if not self.loaded:
                self.init_knn_map()

        self.reload_roll = self.reload_roll + 1

        # self.recent_memory = np.zeros((self.output_shape[1] * self.memory_height, 3), dtype=np.uint8)
        #
        # self.recent_frames = np.zeros(
        #     (self.frame_stacks, self.output_shape[0],
        #      self.output_shape[1], self.output_shape[2]),
        #     dtype=np.uint8)

        self.agent_stats = []

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_shape[:2], fps=60)
            self.model_frame_writer.__enter__()

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.last_opp_health = 1
        self.latest_healing_reward = 0
        self.died_count = 0
        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        self.loaded = True
        return self.render(), {}

    def init_knn_map(self):
        if self.explore_method in ["SCREEN", "HYBRID"]:
            self.init_knn()
        self.init_map_mem()

    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim)  # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)

    def init_map_mem(self):
        self.seen_coords = {}
        self.seen_maps = set([])

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray()  # (144, 160, 3)
        if self.noise:
            noise = np.random.normal(0, 128 * self.noise, size=game_pixels_render.shape)
            np.reshape(noise, game_pixels_render.shape)
            game_pixels_render = game_pixels_render + noise
            game_pixels_render = game_pixels_render.clip(0,255)
        # convert to gray
        # game_pixels_render = np.dot(game_pixels_render[...,:3], [0.299, 0.587, 0.114])
        if reduce_res:
            game_pixels_render = (255 * resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            # if update_mem:
            #     self.recent_frames[0] = game_pixels_render
            # if add_memory:
            #     pad = np.zeros(
            #         shape=(self.mem_padding, self.output_shape[1], 3),
            #         dtype=np.uint8)
            #     game_pixels_render = np.concatenate(
            #         (
            #             self.create_exploration_memory(),
            #             pad,
            #             self.create_recent_memory(),
            #             pad,
            #             rearrange(self.recent_frames, 'f h w c -> (f h) w c')
            #         ),
            #         axis=0)
        return game_pixels_render

    def step(self, action):

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        # self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[
                   frame_start:frame_start + self.output_shape[0], ...].flatten().astype(np.float32)

        if self.explore_method in ["SCREEN", "HYBRID"]:
            self.update_frame_knn_index(obs_flat)
        self.update_seen_coords()

        self.update_heal_reward()

        new_reward, new_prog = self.update_reward()

        self.last_health = self.read_hp_fraction()

        # shift over short term reward memory
        # self.recent_memory = np.roll(self.recent_memory, 3)
        # self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        # self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        # self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward * 0.1, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 16:  # was 8, but then the player just turns, but doesn't step
                if action < 6:
                    # release
                    self.pyboy.send_input(self.release_button[action])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))

    def append_agent_stats(self, action):
        x_pos = self.read_m(self._map_position_x)
        y_pos = self.read_m(self._map_position_y)
        map_n = str(self.read_m(self._map_bank_no)) + "_" + str(self.read_m(self._map_map_no))
        levels = [self.read_m(a) for a in self._pokemon_lvs]
        self.agent_stats.append({
            'step': self.step_count, 'x': x_pos, 'y': y_pos, 'map': map_n,
            'last_action': action,
            'pcount': self.read_m(self._party_total), 'levels': levels, 'ptypes': self.read_party(),
            'hp': self.read_hp_fraction(),
            'deaths': self.died_count, 'badge': self.get_badges(),
            'event': self.progress_reward['event'], 'healr': self.latest_healing_reward
        })

    def update_frame_knn_index(self, frame_vec):
        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current
            labels, distances = self.knn_index.knn_query(frame_vec, k=1)
            if distances[0][0] > self.similar_frame_dist:
                # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )

    def update_seen_coords(self):
        x_pos = self.read_m(self._map_position_x)
        y_pos = self.read_m(self._map_position_y)
        map_n = str(self.read_m(self._map_bank_no)) + "_" + str(self.read_m(self._map_map_no))
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"

        self.seen_coords[coord_string] = self.step_count
        self.seen_maps.add(map_n)

        if not self.init_state and len(self.seen_coords) == 2:
            init_state_file = self.gb_path + ".state"
            with open(init_state_file, "bw") as f:
                self.pyboy.save_state(f)

        if not self.headless:
            folder_name = f"maps_{self.simple_name}"
            folder = self.s_path / Path(folder_name)
            folder.mkdir(exist_ok=True)
            if self.step_count % 200 == 0:
                arr_dict = {}
                for k in self.seen_coords.keys():
                    x, y, m = re.findall(r'[0-9_]+', k)
                    if m not in arr_dict.keys():
                        arr_dict[m] = np.ones((100, 100))
                    arr_dict[m][int(y), int(x)] = 0
                from matplotlib import pyplot as plt
                for m, img in arr_dict.items():
                    crop = True

                    def crop_image(image):
                        if not crop:
                            return image
                        mask = image != 1
                        mask0, mask1 = np.any(mask, 0), np.any(mask, 1)
                        return image[np.ix_(mask1, mask0)]

                    plt.imshow(crop_image(img), cmap="gray")
                    plt.savefig(folder / Path(f"{m}.png"))


    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum(
            [val for _, val in self.progress_reward.items()])  # sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return (new_step,
                (new_prog[0] - old_prog[0],
                 new_prog[1] - old_prog[1],
                 new_prog[2] - old_prog[2])
                )

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (prog['level'] * 100 / self.reward_scale,
                self.read_hp_fraction() * 100,
                prog['explore'] * 150 / (self.explore_weight * self.reward_scale))

    # def create_exploration_memory(self):
    #     w = self.output_shape[1]
    #     h = self.memory_height
    #
    #     def make_reward_channel(r_val):
    #         col_steps = self.col_steps
    #         max_r_val = (w - 1) * h * col_steps
    #         # truncate progress bar. if hitting this
    #         # you should scale down the reward in group_rewards!
    #         r_val = min(r_val, max_r_val)
    #         row = floor(r_val / (h * col_steps))
    #         memory = np.zeros(shape=(h, w), dtype=np.uint8)
    #         memory[:, :row] = 255
    #         row_covered = row * h * col_steps
    #         col = floor((r_val - row_covered) / col_steps)
    #         memory[:col, row] = 255
    #         col_covered = col * col_steps
    #         last_pixel = floor(r_val - row_covered - col_covered)
    #         memory[col, row] = last_pixel * (255 // col_steps)
    #         return memory
    #
    #     level, hp, explore = self.group_rewards()
    #     full_memory = np.stack((
    #         make_reward_channel(level),
    #         make_reward_channel(hp),
    #         make_reward_channel(explore)
    #     ), axis=-1)
    #
    #     if self.get_badges() > 0:
    #         full_memory[:, -1, :] = 255
    #
    #     return full_memory

    # def create_recent_memory(self):
    #     return rearrange(
    #         self.recent_memory,
    #         '(w h) c -> h w c',
    #         h=self.memory_height)

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.total_reward < 30:
                done = True
        else:
            done = self.step_count >= self.max_steps
        # done = self.read_hp_fraction() == 0
        return done

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d} seed:{self.seed} '
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)

        # if self.step_count % 50 == 0:
        #     plt.imsave(
        #         self.s_path / Path(f'curframe_{self.instance_id}.jpeg'),
        #         self.render(reduce_res=False))

        if self.print_rewards and done:
            print('', flush=True)
        if self.save_final_state and done:
            fs_path = self.s_path / Path('final_states')
            fs_path.mkdir(exist_ok=True)
            # plt.imsave(
            #     fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'),
            #     obs_memory)
            # plt.imsave(
            #     fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'),
            #     self.render(reduce_res=False))
            with open(str(self.s_path) + f"/final_states/r{self.total_reward:.4f}_{self.reset_count}.state", "bw") as f:
                self.pyboy.save_state(f)

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == '1'

    def get_levels_sum(self):
        poke_levels = [max(self.read_m(a), 0) for a in self._pokemon_lvs]
        return max(sum(poke_levels), 0)

    def get_levels_reward(self):
        # focus other things over leveling when outscaling enemies too much
        level_sum = self.get_levels_sum()
        return min(level_sum, int(self.max_opponent_level * 6) + 5)

    def get_xp_reward(self):
        return self.read_xp()

    def get_items_reward(self):
        num_items = max(self.read_m(self._num_items), 0)
        num_ball_items = max(self.read_m(self._num_ball_items), 0)
        num_key_items = max(self.read_m(self._num_key_items), 0)
        return sum([num_items * 20, num_ball_items * 5, num_key_items * 10])

    def get_explore_reward(self):
        bonus_reward_maps = ["24_3"]  # next route
        bonus_reward = 2
        low_reward_maps = ["24_9", "24_8", "24_6", "24_7"]  # houses in new bark town
        low_reward = 1
        steps = len(self.seen_coords)
        for m in bonus_reward_maps:
            steps += len([x for x in self.seen_coords.keys() if f"m:{m}" in x]) * (bonus_reward - 1)
        for m in low_reward_maps:
            steps -= len([x for x in self.seen_coords.keys() if f"m:{m}" in x]) * (1 - low_reward)
        if self.explore_method == "STEPS":
            return steps
        pre_rew = 0.005
        post_rew = 0.01
        cur_size = self.knn_index.get_current_count()
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        screen = base + post
        if self.explore_method == "SCREEN":
            return screen
        if self.explore_method == "HYBRID":
            return steps + screen

    def get_badges(self):
        return self.bit_count(self.read_m(self._badges))

    def get_hms(self):
        return self.bit_count(self.read_m(self._hms))

    def get_seen_count(self):
        return self.read_pokedex_count(self._pokedex_seen_from, self._pokedex_seen_to)

    def get_caught_count(self):
        return self.read_pokedex_count(self._pokedex_own_from, self._pokedex_own_to)

    def get_maps_explored(self):
        return len(self.seen_maps) if len(self.seen_maps) < 8 else len(self.seen_maps) * 2

    def multiply_map_steps(self):
        total = 1
        arr_dict = {}
        for k in self.seen_coords.keys():
            x, y, m = re.findall(r'[0-9_]+', k)
            if m not in arr_dict.keys():
                arr_dict[m] = 0
            arr_dict[m] = arr_dict[m] + 1
        for v in arr_dict.values():
            total *= v
        return total - 1

    def read_party(self):
        return [self.read_m(addr) for addr in self._party_pokemon]

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.latest_healing_reward = heal_amount
            else:
                self.died_count += 1

    def get_damage_reward(self):
        curr_opp_health = self.read_opp_hp_fraction()
        if self.get_levels_sum() <= self.get_levels_reward():
            if curr_opp_health <= self.last_opp_health:
                rew = self.last_opp_health - curr_opp_health
                self.last_opp_health = curr_opp_health
                return rew
            else:
                self.last_opp_health = curr_opp_health
        return 0

    def get_all_events_reward(self):
        return max(sum([self.bit_count(self.read_bit(i, 1)) for i in self._event_flags]), 0)

    def get_game_state_reward(self):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        state_scores = {
            'event': self.reward_scale * self.update_max_event_reward(),
            'level': self.reward_scale * self.get_levels_reward() * 10,
            'xp': self.reward_scale * self.get_xp_reward(),
            'items': self.reward_scale * self.get_items_reward(),
            'heal': self.reward_scale * self.latest_healing_reward,
            'op_lvl': self.reward_scale * self.update_max_op_level(),
            'op_dmg': self.reward_scale * self.get_damage_reward() * 10,
            # 'dead': self.reward_scale * -1.0 * self.died_count,
            'badge': self.reward_scale * self.get_badges() * 5,
            # 'hms': self.reward_scale * self.get_hms() * 5,
            # 'money': self.reward_scale* money * 3,
            'seen_count': self.reward_scale * self.get_seen_count(),
            'caught_count': self.reward_scale * self.get_caught_count(),
            'explore': self.reward_scale * self.explore_weight * self.get_explore_reward() * 2,
            'map_explore': self.reward_scale * self.get_maps_explored() * 10,
            # 'overvalue_new_maps': self.multiply_map_steps() * 0.01,
            'neg_steps': -0.01
        }

        return state_scores

    def save_screenshot(self, name):
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg'),
            self.render(reduce_res=False))

    def update_max_op_level(self):
        opponent_level = self.read_m(self._opponent_level)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_reward(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def read_hp_fraction(self):
        hp_sum = sum([self.read_hp(hp) for hp in self._pokemon_hps])
        max_hp_sum = sum([self.read_hp(hp) for hp in self._pokemon_max_hps])
        if hp_sum == 0 or max_hp_sum == 0:
            return 0
        return hp_sum / max_hp_sum

    def read_opp_hp_fraction(self):
        hp = self.read_hp(self._opponent_stats)
        max_hp = self.read_hp(self._opponent_stats + 2)
        if max_hp == 0:
            return 1
        return hp / max_hp

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256 * 256 * self.read_m(start_add) + 256 * self.read_m(start_add + 1) + self.read_m(start_add + 2)

    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

    def read_money(self):
        return self.read_3bcd(self._money)

    def read_xp(self):
        return sum([self.read_3bcd(a) for a in self._pokemon_xps])

    def read_3bcd(self, base):
        return (100 * 100 * self.read_bcd(self.read_m(base)) +
                100 * self.read_bcd(self.read_m(base + 1)) +
                self.read_bcd(self.read_m(base + 2)))

    def read_pokedex_count(self, start, end):
        return sum([self.bit_count(self.read_bit(i, 1)) for i in range(start, end + 1)])
