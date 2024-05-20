import random
import sys
import uuid
import os
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
import pyboy.botsupport.screen
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib
import mediapy as media
import pandas as pd

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent


class GoldGymEnv(Env):
    _map_position_x = 0xD20D
    _map_position_y = 0xD20E
    _map_bank_no = 0xDA00
    _map_map_no = 0xDA01

    _party_total = 0xDA22
    _party1 = 0xDA23
    _party2 = 0xDA24
    _party3 = 0xDA25
    _party4 = 0xDA26
    _party5 = 0xDA27
    _party6 = 0xDA28
    _party_pokemon = [_party1, _party2, _party3, _party4, _party5, _party6]

    _pokemon1_lv = 0xDA49
    _pokemon1_hp = 0xDA4C
    _pokemon1_max_hp = 0xDA4E
    _pokemon1_xp = 0xDA32

    _pokemon2_lv = 0xDA79
    _pokemon2_hp = 0xDA7C
    _pokemon2_max_hp = 0xDA7E
    _pokemon2_xp = 0xDA62

    _pokemon3_lv = 0xDAA9
    _pokemon3_hp = 0xDAAC
    _pokemon3_max_hp = 0xDAAE
    _pokemon3_xp = 0xDA92

    _pokemon4_lv = 0xDAD9
    _pokemon4_hp = 0xDADC
    _pokemon4_max_hp = 0xDADE
    _pokemon4_xp = 0xDAC2

    _pokemon5_lv = 0xDB09
    _pokemon5_hp = 0xDB0C
    _pokemon5_max_hp = 0xDB0E
    _pokemon5_xp = 0xDAF2

    _pokemon6_lv = 0xDB39
    _pokemon6_hp = 0xDB3C
    _pokemon6_max_hp = 0xDB3E
    _pokemon6_xp = 0xDB22

    _pokemon_lvs = [_pokemon1_lv, _pokemon2_lv, _pokemon3_lv, _pokemon4_lv, _pokemon5_lv, _pokemon6_lv]
    _pokemon_hps = [_pokemon1_hp, _pokemon2_hp, _pokemon3_hp, _pokemon4_hp, _pokemon5_hp, _pokemon6_hp]
    _pokemon_max_hps = [_pokemon1_max_hp, _pokemon2_max_hp, _pokemon3_max_hp, _pokemon4_max_hp, _pokemon5_max_hp,
                        _pokemon6_max_hp]
    _pokemon_xps = [_pokemon1_xp, _pokemon2_xp, _pokemon3_xp, _pokemon4_xp, _pokemon5_xp, _pokemon6_xp]

    _num_items = 0xD5B7
    _num_ball_items = 0xD5FC
    _num_key_items = 0xD5E1

    # badges are hex values, 01, 02, 04, 08, 10, 20, 40, 80. totalling FF for all badges.
    _badges = 0xD57C
    _hms = 0xD5B0

    _opponent_level = 0xD0FC
    _opponent_stats = 0xD0FF  # Stats of current opponent. Each value is two-byte big-endian in the following order: current HP, total HP, Attack, Defense, Speed, Sp. Atk., Sp. Def.

    _money = 0xD573

    _pokedex_own_from = 0xDBE4
    _pokedex_own_to = 0xDC03
    _pokedex_seen_from = 0xDC04
    _pokedex_seen_to = 0xDC23

    _event_flags = [0xD67C,  # Pokegear
                    0xBD06,  # = Player has Pokédex
                    0xBE06,  # = Rival has stolen Pokémon
                    0xC106,  # = Met Rival in Goldenrod Underground
                    0xC206,  # = Met Rival in Cherrygrove
                    0xC306,  # = Olivine Gym Leader in Lighthouse
                    0xC406,  # = Met Rival in Sprout Tower
                    0xC506,  # = Met Rival in Burned Tower
                    0xC706,  # = Player comes down 1st time
                    0xC806,  # = Player has Pokémon
                    0xC906,  # = 1st time in Mr. Pokémon House
                    0xCB06,  # = Teacher in school
                    0xCC06,  # = TR left Goldenrod
                    0xCE06,  # = TR has attacked Radio Tower
                    0xD006,  # = TR attacked Radio Tower Once, but they are gone.
                    0xD306,  # = Lighthouse Pokémon cured
                    0xD406,  # = Battled Red Gyarados
                    0xD506,  # = Lance is in Mahogany Store
                    0xD606,  # = Lance in B2
                    0xD706,  # = Player defeated Final TR in Slowpoke Well
                    0xD806,  # = Player got Dragon's Den Item
                    0xD906,  # = Object Event: Team Rocket in B1
                    0xDB06,  # = Beat Team Rocket Executive
                    0xDC06,  # = TR is in Mahogany
                    0xDF06,  # = Lance hurting other guy
                    0xE006,  # = Voltorb 1 in Mahogany fainted
                    0xE106,  # = Voltorb 2 in Mahogany fainted
                    0xE206,  # = Voltorb 3 in Mahogany fainted
                    0xE406,  # = Got Team Rocket out of Goldenrod
                    0xE906,  # = Farfetch'd Position 1
                    0xEA06,  # = Farfetch'd Position 2
                    0xEB06,  # = Farfetch'd Position 3
                    0xEC06,  # = Farfetch'd Position 4
                    0xED06,  # = Farfetch'd Position 5
                    0xEE06,  # = Farfetch'd Position 6a
                    0xEF06,  # = Farfetch'd Position 6b
                    0xF006,  # = Farfetch'd Position 7
                    0xF106,  # = Farfetch'd Position 8
                    0xF206,  # = Farfetch'd Position (End)
                    0xF406,  # = Farfetch'd brought back
                    0xF806,  # = Player battled Sudowoodo
                    0xFA06,  # = TR is in Azalea
                    0xFF06,  # = Guide Gent has given map
                    0x0007]  # = On SS Aqua for first time

    def __init__(
            self, config=None):

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320  # 1000
        self.headless = config['headless']
        self.num_elements = 20000  # max
        self.init_state = None if 'init_state' not in config else config['init_state']
        self.load_once = False if 'load_once' not in config else config['load_once']
        self.loaded = False
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.similar_frame_dist = config['sim_frame_dist']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (-10, 100)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PASS
            ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(100)

        # self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        else:
            seed = random.randint(0, 10000)
            # print("reshuffling seed", seed)
            self.seed = seed

        # restart game, skipping credits
        if self.init_state:
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
            if self.load_once and not self.loaded:
                self.init_state = None

        if not self.loaded:
            if self.use_screen_explore:
                self.init_knn()
            self.init_map_mem()

        self.recent_memory = np.zeros((self.output_shape[1] * self.memory_height, 3), dtype=np.uint8)

        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0],
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

        self.agent_stats = []

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()

        self.levels_satisfied = False
        self.levels_satisfied_min = 5
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.last_opp_health = 1
        self.total_healing_reward = 0
        self.total_damage_reward = 0
        self.died_count = 0
        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        self.loaded = True
        return self.render(), {}

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
        if reduce_res:
            game_pixels_render = (255 * resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3),
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(),
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render

    def step(self, action):

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[
                   frame_start:frame_start + self.output_shape[0], ...].flatten().astype(np.float32)

        if self.use_screen_explore:
            self.update_frame_knn_index(obs_flat)
        self.update_seen_coords()

        self.update_heal_reward()

        new_reward, new_prog = self.update_reward()

        self.last_health = self.read_hp_fraction()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

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
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if 3 < action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
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
        if self.use_screen_explore:
            expl = ('frames', self.knn_index.get_current_count())
        else:
            expl = ('coord_count', len(self.seen_coords))
        self.agent_stats.append({
            'step': self.step_count, 'x': x_pos, 'y': y_pos, 'map': map_n,
            'last_action': action,
            'pcount': self.read_m(self._party_total), 'levels': levels, 'ptypes': self.read_party(),
            'hp': self.read_hp_fraction(),
            expl[0]: expl[1],
            'deaths': self.died_count, 'badge': self.get_badges(),
            'event': self.progress_reward['event'], 'healr': self.total_healing_reward
        })

    def update_frame_knn_index(self, frame_vec):

        if self.get_levels_sum() >= self.levels_satisfied_min and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

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
        if self.get_levels_sum() >= self.levels_satisfied_min and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}

        self.seen_coords[coord_string] = self.step_count
        self.seen_maps.add(map_n)

    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        old = self.progress_reward.copy()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum(
            [val for _, val in self.progress_reward.items()])  # sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        # if new_step < 0 and self.read_hp_fraction() > 0:
        # print(f'\n\nreward went down! instance: {self.instance_id}\n{old}\n{self.progress_reward}\n\n')
        # self.save_screenshot('neg_reward')

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
        # (prog['events'],
        # prog['levels'] + prog['party_xp'],
        # prog['explore'])

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height

        def make_reward_channel(r_val):
            col_steps = self.col_steps
            max_r_val = (w - 1) * h * col_steps
            # truncate progress bar. if hitting this
            # you should scale down the reward in group_rewards!
            r_val = min(r_val, max_r_val)
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered)
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory

        level, hp, explore = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)

        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self):
        return rearrange(
            self.recent_memory,
            '(w h) c -> h w c',
            h=self.memory_height)

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
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
            if self.save_final_state:
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
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')

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
        return sum([num_items * 0.05, num_ball_items * 0.1, num_key_items * 2])

    def get_explore_reward(self):
        if not self.use_screen_explore:
            return len(self.seen_coords) * 0.1

        pre_rew = 0.005
        post_rew = 0.01
        cur_size = self.knn_index.get_current_count()
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post

    def get_badges(self):
        return self.bit_count(self.read_m(self._badges))

    def get_hms(self):
        return self.bit_count(self.read_m(self._hms))

    def get_seen_count(self):
        return self.read_pokedex_count(self._pokedex_seen_from, self._pokedex_seen_to)

    def get_caught_count(self):
        return self.read_pokedex_count(self._pokedex_own_from, self._pokedex_own_to)

    def get_maps_explored(self):
        return len(self.seen_maps)

    def read_party(self):
        return [self.read_m(addr) for addr in self._party_pokemon]

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:  # exclude levelups
                    print(f'healed: {heal_amount}')
                    self.save_screenshot('healing')
                self.total_healing_reward += heal_amount
            else:
                self.died_count += 1

    def get_damage_reward(self):
        curr_opp_health = self.read_opp_hp_fraction()
        if self.get_levels_sum() <= self.get_levels_reward():
            if curr_opp_health <= self.last_opp_health:
                rew = self.last_opp_health - curr_opp_health
                self.last_opp_health = curr_opp_health
                self.total_damage_reward += rew
            else:
                self.last_opp_health = curr_opp_health
        return self.total_damage_reward

    def get_all_events_reward(self):
        # flags = [hex(i) for i in self._event_flags]
        # values = [self.bit_count(self.read_bit(i, 1)) for i in self._event_flags]
        # print("\n", list(zip(flags, values)), sum([self.bit_count(self.read_bit(i, 1)) for i in self._event_flags]))
        return max(sum([self.bit_count(self.read_bit(i, 1)) for i in self._event_flags]), 0)

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        state_scores = {
            'event': self.reward_scale * (self.update_max_event_reward() ** 2) * 0.01,
            'level': self.reward_scale * self.get_levels_reward(),
            # 'xp': self.reward_scale * self.get_xp_reward() * 0.1,
            'items': self.reward_scale * self.get_items_reward(),
            'heal': self.reward_scale * self.total_healing_reward,
            'op_lvl': self.reward_scale * self.update_max_op_level(),
            'op_dmg': self.reward_scale * self.get_damage_reward(),
            'dead': self.reward_scale * -1.0 * self.died_count,
            'badge': self.reward_scale * self.get_badges() * 5,
            'hms': self.reward_scale * self.get_hms() * 5,
            # 'money': self.reward_scale* money * 3,
            'seen_count': self.reward_scale * self.get_seen_count() * 0.01,
            'caught_count': self.reward_scale * self.get_caught_count() * 0.1,
            'explore': self.reward_scale * self.explore_weight * self.get_explore_reward(),
            'map_explore': self.reward_scale * self.get_maps_explored(),
            'neg_steps': self.step_count * -0.001
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
        return self.max_opponent_level * 0.4

    def update_max_event_reward(self):
        cur_rew = self.get_all_events_reward()
        # if cur_rew - self.max_event_rew > 10:
        #     print(f"\nhit event: {cur_rew}. resetting map mem")
        #     self.init_map_mem()
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
