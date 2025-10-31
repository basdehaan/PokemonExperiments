from training.pokemon_env import PokeGymEnv


class RedGymEnv(PokeGymEnv):
    gb_path = '../PokemonRed.gb'
    simple_name = "red"
    version_indicator = 1

    _map_position_x = 0xD362
    _map_position_y = 0xD361
    _map_bank_no = 0xD35D
    _map_map_no = 0xD35E

    _party_total = 0xD163
    _party1 = 0xD164
    _party2 = 0xD165
    _party3 = 0xD166
    _party4 = 0xD167
    _party5 = 0xD168
    _party6 = 0xD169

    _party_pokemon = [_party1, _party2, _party3, _party4, _party5, _party6]

    _pokemon1_lv = 0xD18C
    _pokemon1_hp = 0xD16C
    _pokemon1_max_hp = 0xD18D
    _pokemon1_xp = 0xD179

    _pokemon2_lv = 0xD1B8
    _pokemon2_hp = 0xD198
    _pokemon2_max_hp = 0xD1B9
    _pokemon2_xp = 0xD1A5

    _pokemon3_lv = 0xD1E4
    _pokemon3_hp = 0xD1C4
    _pokemon3_max_hp = 0xD1E5
    _pokemon3_xp = 0xD1D1

    _pokemon4_lv = 0xD210
    _pokemon4_hp = 0xD1F0
    _pokemon4_max_hp = 0xD211
    _pokemon4_xp = 0xD1FD

    _pokemon5_lv = 0xD23C
    _pokemon5_hp = 0xD21C
    _pokemon5_max_hp = 0xD23D
    _pokemon5_xp = 0xD229

    _pokemon6_lv = 0xD268
    _pokemon6_hp = 0xD248
    _pokemon6_max_hp = 0xD269
    _pokemon6_xp = 0xD255

    _pokemon_lvs = [_pokemon1_lv, _pokemon2_lv, _pokemon3_lv, _pokemon4_lv, _pokemon5_lv, _pokemon6_lv]
    _pokemon_hps = [_pokemon1_hp, _pokemon2_hp, _pokemon3_hp, _pokemon4_hp, _pokemon5_hp, _pokemon6_hp]
    _pokemon_max_hps = [_pokemon1_max_hp, _pokemon2_max_hp, _pokemon3_max_hp, _pokemon4_max_hp, _pokemon5_max_hp,
                        _pokemon6_max_hp]
    _pokemon_xps = [_pokemon1_xp, _pokemon2_xp, _pokemon3_xp, _pokemon4_xp, _pokemon5_xp, _pokemon6_xp]

    _num_items = 0xD53A

    _badges = 0xD356

    _opponent_level = 0xCFE8
    _opponent_stats = 0xD002
    _opponent_party_levels = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]

    _money = 0xD347

    _pokedex_own_from = 0xD2F7
    _pokedex_own_to = 0xD309
    _pokedex_seen_from = 0xD30A
    _pokedex_seen_to = 0xD31C

    _event_flags_start = 0xD747
    _event_flags_end = 0xD886
    _museum_ticket = 0xD754
    _event_flags = [i for i in range(_event_flags_start, _event_flags_end)]

    def __init__(
            self, config=None):
        super().__init__(config)

    def get_hms(self):
        return 0

    def get_items_reward(self):
        num_items = max(self.read_m(self._num_items), 0)
        return num_items * 20
