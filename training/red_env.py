from training.pokemon_env import PokeGymEnv


class RedGymEnv(PokeGymEnv):
    _map_position_x = 0xD362
    _map_position_y = 0xD361
    _map_bank_no = 0xDA00
    _map_map_no = 0xD35E

    _party_pokemon = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]

    _pokemon_lvs = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
    _pokemon_hps = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
    _pokemon_max_hps = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
    _money = 0xD347
    _opponent_level = 0xCFE8
    _opponent_party_levels = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]

    _event_flags_start = 0xD747
    _event_flags_end = 0xD886
    _museum_ticket = 0xD754
    _event_flags = [i for i in range(_event_flags_start, _event_flags_end)].append(_museum_ticket)

    _badges = 0xD356

    def __init__(
            self, config=None):
        super().__init__(config)
