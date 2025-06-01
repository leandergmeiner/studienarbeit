import vizdoom as vzd
from typing import SupportsFloat

DOOM_BUTTONS = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.ATTACK,
    vzd.Button.SPEED,
    vzd.Button.CROUCH,
]


def delta(factor: SupportsFloat):
    def _delta_func(new: SupportsFloat, old: SupportsFloat):
        return factor * (new - old)

    return _delta_func


# Leaky ReLU
def leru(f1: SupportsFloat, f2: SupportsFloat):
    def _leru_func(new: SupportsFloat, old: SupportsFloat):
        d = new - old
        return f1 * min(0, d) + f2 * max(0, d)

    return _leru_func


# TODO: Arnold Reward Functions
DEFAULT_REWARD_FUNCS = {
    vzd.GameVariable.DEATHCOUNT: delta(-5000),
    vzd.GameVariable.KILLCOUNT: delta(1000),
    vzd.GameVariable.ITEMCOUNT: delta(100),
    vzd.GameVariable.ARMOR: delta(10),
    vzd.GameVariable.HEALTH: delta(10),
    vzd.GameVariable.DAMAGE_TAKEN: delta(-10),
    vzd.GameVariable.DAMAGECOUNT: delta(30),
    vzd.GameVariable.SELECTED_WEAPON_AMMO: leru(1, 10),
}

DEFEND_CENTER_REWARD_FUNCS = {    
    vzd.GameVariable.DEATHCOUNT: delta(0),
    vzd.GameVariable.KILLCOUNT: delta(2),
    vzd.GameVariable.ITEMCOUNT: delta(0),
    vzd.GameVariable.ARMOR: delta(0),
    vzd.GameVariable.HEALTH: delta(0),
    vzd.GameVariable.DAMAGE_TAKEN: delta(-1),
    vzd.GameVariable.DAMAGECOUNT: delta(0),
    vzd.GameVariable.SELECTED_WEAPON_AMMO: delta(0),
}

HEALTH_GATHERING_REWARD_FUNCS = {    
    vzd.GameVariable.DEATHCOUNT: delta(0),
    vzd.GameVariable.KILLCOUNT: delta(0),
    vzd.GameVariable.ITEMCOUNT: delta(0),
    vzd.GameVariable.ARMOR: delta(0),
    vzd.GameVariable.HEALTH: delta(0),
    vzd.GameVariable.DAMAGE_TAKEN: delta(0),
    vzd.GameVariable.DAMAGECOUNT: delta(0),
    vzd.GameVariable.SELECTED_WEAPON_AMMO: delta(0),
}