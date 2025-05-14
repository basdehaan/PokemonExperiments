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

from gymnasium import spaces
from pyboy.utils import WindowEvent

from training.pokemon_env import PokeGymEnv


class GoldGymEnv(PokeGymEnv):
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
        super().__init__(config)





    def update_seen_coords(self):
        super().update_seen_coords()
        # Maps:
        # lab: 24_5
        # new bark town: 24_4
        # new bark house bottom right: 24_9
        # new bark house bottom left: 24_8
        # home: 24_6
        # home upstairs: 24_7
        # route 29: 24_3
        if not self.headless:
            if self.step_count % 200 == 0:
                arr_dict = {}
                for k in self.seen_coords.keys():
                    x, y, m = re.findall(r'[0-9_]+', k)
                    if m not in arr_dict.keys():
                        arr_dict[m] = np.ones((100, 100))
                    arr_dict[m][int(y), int(x)] = 0
                from matplotlib import pyplot as plt
                (self.s_path / Path("maps")).mkdir(exist_ok=True)
                for m, img in arr_dict.items():
                    crop = True
                    def crop_image(image):
                        if not crop:
                            return image
                        mask = image != 1
                        mask0, mask1 = np.any(mask, 0), np.any(mask, 1)
                        return image[np.ix_(mask1, mask0)]

                    plt.imshow(crop_image(img), cmap="gray")
                    plt.savefig(self.s_path / Path("maps") / Path(f"{m}.png"))
