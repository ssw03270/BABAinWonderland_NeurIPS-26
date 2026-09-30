from dataclasses import dataclass
from functools import partial
from typing import Iterable
from gym import spaces
import numpy as np

from baba_in_wonderland.grid import BabaIsYouGrid, BabaIsYouEnv, put_rule, place_rule
from baba_in_wonderland.world_object import Baba, RuleAnd, RuleIs, RuleObject, RuleProperty, Wall, make_obj
from baba_in_wonderland import make, register
from src.environments.custom_map_spec import (
    ASCII_OBJECTS as CUSTOM_ASCII_OBJECTS,
    ASCII_RULE_OPERATORS as CUSTOM_ASCII_RULE_OPERATORS,
    ASCII_RULE_OBJECTS as CUSTOM_ASCII_RULE_OBJECTS,
    ASCII_RULE_PROPERTIES as CUSTOM_ASCII_RULE_PROPERTIES,
    CUSTOM_MAP_ROOT,
    custom_map_has_fixed_dimensions,
    get_custom_map_dimensions,
    load_custom_map_catalog,
)



def put_obj(env, obj, pos):
    """
    Put an object at a position in the grid.
    Args:
        env: environment
        obj: object to put, tuple (color, name) or string name
        pos: position to put the object, (i, j)
    """
    if isinstance(obj, tuple):
        color, obj = obj
    else:
        color = None
    obj = make_obj(obj, color=color) if isinstance(obj, str) else obj
    env.put_obj(obj, *pos)


def make_rule_operator_block(word: str):
    normalized = str(word).strip().lower()
    if normalized == "is":
        return RuleIs()
    if normalized == "and":
        return RuleAnd()
    raise ValueError(f"Unsupported rule operator `{word}`.")


def place_obj(env, obj, top=None, size=None):
    """
    Place an object at a random empty position in the grid.
    Args:
        env: environment
        obj: object to place, tuple (color, name) or string name
        top: top left position of the area to place the object
        size: size of the area to place the object
    """
    if isinstance(obj, tuple):
        color, obj = obj
    else:
        color = None
    obj = make_obj(obj, color=color) if isinstance(obj, str) else obj
    pos = env.place_obj(obj, top=top, size=size)
    return pos


def break_rule(env, rule, new_pos={}, block_idx=None):
    """
    Break a rule by moving one of its blocks to a new position.
    Args:
        rule: rule to break
        new_pos: new position for the block
        block_idx: index of the block to move
    """
    rule_pos = env.init_rules[rule]

    if isinstance(new_pos, dict):
        new_pos = env.place_obj(None, **new_pos)  # pos constraints the new_pos for the rule block

    if block_idx is None:
        # pick one block of the rule and displace it
        block_idx = np.random.choice(len(rule_pos))
    elif isinstance(block_idx, Iterable):
        block_idx = block_idx[np.random.choice(len(block_idx))]

    p = rule_pos[block_idx]
    env.change_obj_pos(p, new_pos)
    # env.solution[rule] = {
    #     "push": (new_pos, p)
    # }


NAMES = ["crab", "key", "door"]
COLORS = ["red", "blue", "green"]
OBJECTS = [
    (color, name) for color in COLORS for name in NAMES
]
ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}

@register("env/you_win")
class YouWinEnv(BabaIsYouEnv):
    def __init__(self, width=6, height=6, fixed_you=False, **kwargs):
        self.fixed_you = fixed_you
        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        # randomly sample the parameters
        indices = np.random.choice(len(OBJECTS), size=4, replace=False)
        win_obj, you_obj, distractor1, distractor2 = [OBJECTS[i] for i in indices]

        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        put_obj(self, Wall(), (1, 3))
        put_obj(self, Wall(), (1, 4))
        put_obj(self, Wall(), (4, 3))
        put_obj(self, Wall(), (4, 4))

        if not self.fixed_you:
            # randomly sample the positions
            all_pos = [(2, 3), (3, 3), (2, 4), (3, 4)]
            you_obj_pos = all_pos.pop(np.random.choice(4))
        else:
            you_obj_pos = (2, 3)
            all_pos = [(2, 4), (3, 3), (3, 4)]

        # make sure the win object is one cell away from the you object
        # e.g. if you object is at (2, 3), win object can be at (2, 4) or (3, 3)
        while True:
            idx = np.random.choice(3)
            win_obj_pos = all_pos[idx]
            if abs(win_obj_pos[0] - you_obj_pos[0]) + abs(win_obj_pos[1] - you_obj_pos[1]) == 1:
                break
        all_pos.pop(idx)
        distractor1_pos = all_pos.pop(np.random.choice(2))
        distractor2_pos = all_pos[0]

        put_rule(self, you_obj, "you", positions=(1, 1))
        put_rule(self, win_obj, "win", positions=(1, 2))
        put_obj(self, you_obj, you_obj_pos)
        put_obj(self, win_obj, win_obj_pos)
        put_obj(self, distractor1, distractor1_pos)
        put_obj(self, distractor2, distractor2_pos)

        # action to move the you object to the win object
        target_action = None
        for action, (dx, dy) in ACTIONS.items():
            if (you_obj_pos[0] + dx, you_obj_pos[1] + dy) == win_obj_pos:
                target_action = action
                break
        self.target_action = target_action

        self.objects = {
            you_obj: you_obj_pos,
            win_obj: win_obj_pos,
            distractor1: distractor1_pos,
            distractor2: distractor2_pos
        }

        self.active_rules = [
            f"{you_obj[0]} {you_obj[1]} is you",
            f"{win_obj[0]} {win_obj[1]} is win"
        ]

        self.target_plan = f"goto[{win_obj}]"


@register("env/you_win-fixed_you")
class YouWinFixedYouEnv(YouWinEnv):
    def __init__(self, **kwargs):
        super().__init__(fixed_you=True, **kwargs)


@register("env/make_win-distr_obj_rule")
class MakeWinEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=8,
            height=8,
            color_in_rule=False,
            break_win_rule=True,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=False,
            distractor_win_rule=False,
            win_obj_set=None,
            **kwargs
        ):
        self.color_in_rule = color_in_rule
        self.break_win_rule = break_win_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor
        # add a distractor active win rule (not just a rule block for the distractor object)
        self.distractor_win_rule = distractor_win_rule

        self.win_obj_set = win_obj_set if win_obj_set is not None else NAMES

        super().__init__(width=width, height=height, **kwargs)

    def _sample_objects(self):
            # Sample the objects
        if self.color_in_rule:
            # obj1 and obj2 can be of the same type but different colors
            obj1_idx, obj2_idx, obj3_idx = np.random.choice(len(OBJECTS), size=3, replace=False)
            obj1 = OBJECTS[obj1_idx]
            obj2 = OBJECTS[obj2_idx]
            # obj3 only useful when distractor_rule_block is True and irrelevant_rule_distractor is True
            obj3 = OBJECTS[obj3_idx]
        else:
            # make sure obj1 and obj2 are different object types
            obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
            obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
            obj1 = (COLORS[obj1_color], NAMES[obj1_name])
            obj2 = (COLORS[obj2_color], NAMES[obj2_name])
            obj3 = (COLORS[obj3_color], NAMES[obj3_name])
        return obj1, obj2, obj3

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Sample the objects
        obj1, obj2, obj3 = self._sample_objects()
        # TODO: constraint only object name (don't support color constraint yet)
        while obj1[1] not in self.win_obj_set:
            obj1, obj2, obj3 = self._sample_objects()

        # Add the rules
        put_rule(self, "baba", "you", positions=(1, 6))

        if self.color_in_rule:
            win_obj = obj1
            block_idx = 1
        else:
            win_obj = obj1[1]
            block_idx = 0
        # add win rule
        put_rule(self, win_obj, "win", positions=(1, 1))
        # break the win rule
        if self.break_win_rule:
            break_rule(self, (win_obj, "win"), new_pos={"top": (2, 2), "size": (4, 4)}, block_idx=block_idx)

        if self.distractor_rule_block:
            # add distractor rule block for the other object
            # but place it such that it is impossible to make the obj2 win
            positions = [
                (4, 6), (5, 6), (6, 6),  # bottom row
                (4, 1), (5, 1), (6, 1),  # top row
                (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)   # right column
            ]
            pos = positions[np.random.choice(len(positions))]

            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                put_obj(self, RuleObject(obj3[1]), pos)
            else:
                put_obj(self, RuleObject(obj2[1]), pos)

        # Place the objects and agent in the grid
        place_obj(self, "baba", top=(1, 2))
        place_obj(self, obj1, top=(1, 2))
        if self.distractor_obj:
            place_obj(self, obj2, top=(1, 2))

        # add extra distractor win rule
        if self.distractor_win_rule:
            put_rule(self, obj3[1], "win", positions=(4, 4))

        if self.color_in_rule:
            self.win_rule = f"{win_obj[0]} {win_obj[1]} is win"
        else:
            self.win_rule = f"{win_obj} is win"
        self.win_obj = win_obj

        if self.break_win_rule:
            self.target_plan = f"make[{self.win_rule}], goto[{self.win_obj}]" if self.color_in_rule else f"make[{self.win_rule}], goto[{self.win_obj}]"
        else:
            self.target_plan = f"goto[{self.win_obj}]"


@register("env/goto_win-distr_obj_rule")
class GotoWinEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, **kwargs)

@register("env/goto_win")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=False, distractor_rule_block=False, **kwargs)

@register("env/goto_win-distr_obj")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=True, distractor_rule_block=False, **kwargs)

@register("env/goto_win-distr_rule")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=False, distractor_rule_block=True, **kwargs)

@register("env/goto_win-distr_obj-irrelevant_rule")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/goto_win-distr_win_rule")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=True, distractor_win_rule=True, **kwargs)


@register("env/make_win-distr_obj")
class MakeWinNoDistractorRuleEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_rule_block=False, **kwargs)

@register("env/make_win-distr_rule")
class MakeWinNoDistractorObjEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_obj=False, **kwargs)

@register("env/make_win")
class MakeWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_obj=False, distractor_rule_block=False, **kwargs)

@register("env/make_win-distr_obj-irrelevant_rule")
class MakeWinIrrelevantDistractorRuleEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


# ===== Single-room make win splits =====
env_ids = [
    "env/make_win",
    "env/make_win-distr_obj",
    "env/make_win-distr_rule",
    "env/make_win-distr_obj-irrelevant_rule",
    "env/make_win-distr_obj_rule",
]
for env_id in env_ids:
    # NAMES minus "crab"
    win_obj_set = [name for name in NAMES if name != "crab"]
    register(
        f"{env_id}#no_crab_win",
        partial(make, env_id, win_obj_set=win_obj_set)
    )
    register(
        f"{env_id}#only_crab_win",
        partial(make, env_id, win_obj_set=["crab"]),
    )



class TwoRoomEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=13,
            height=9,
            baba_pos="left_pushable",
            obj1_pos="anywhere",
            obj2_pos="anywhere",
            break_stop_rule=False,
            break_win_rule=False,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=False,
            color_in_rule=False,
            distractor_win_rule=False,
            **kwargs
        ):
        self.color_in_rule = color_in_rule
        self.break_stop_rule = break_stop_rule
        self.break_win_rule = break_win_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor
        self.distractor_win_rule = distractor_win_rule

        pushable_area_size = (width // 2 - 2, height - 6)
        self.positions = {
            # not all left or right space because want it to be possible to push the objects
            "left_pushable": {
                "top": (2, 3),
                "size": pushable_area_size
            },
            "right_pushable": {
                "top": (width // 2 + 1, 3),
                "size": pushable_area_size
            },
            "right_unpushable": [
                # left border
                *[(width-2, 2+i) for i in range(height-4)],
                # bottom border
                *[(width-2-i, height-2) for i in range(width//2-2)],
            ],
            "left_anywhere": {
                "top": (1, 1),
                "size": (width // 2 - 2, height - 2)
            },
            "right_anywhere": {
                # start at y = 2 to prevent having object at the same place as the obj rule block of the win rule for the make_win envs
                "top": (width // 2 + 1, 2),
                "size": (width // 2 - 2, height - 2)
            },
            "anywhere": {
                "top": (1, 2),
                "size": (width - 2, height - 2)
            }
        }
        self.left_pushable = self.positions["left_pushable"]
        self.right_pushable = self.positions["right_pushable"]

        self.baba_pos = self.positions[baba_pos]
        self.obj1_pos = self.positions[obj1_pos]
        self.obj2_pos = self.positions[obj2_pos]

        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add the vertical wall dividing the two rooms
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))

        # Sample the objects
        if self.color_in_rule:
            # obj1 and obj2 can be of the same type but different colors
            obj1_idx, obj2_idx, obj3_idx = np.random.choice(len(OBJECTS), size=3, replace=False)
            obj1 = OBJECTS[obj1_idx]
            obj2 = OBJECTS[obj2_idx]
            # obj3 only useful when distractor_rule_block is True and irrelevant_rule_distractor is True
            obj3 = OBJECTS[obj3_idx]
        else:
            # make sure obj1 and obj2 are different object types
            obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
            obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
            obj1 = (COLORS[obj1_color], NAMES[obj1_name])
            obj2 = (COLORS[obj2_color], NAMES[obj2_name])
            obj3 = (COLORS[obj3_color], NAMES[obj3_name])

        # Add the rules
        # put "baba is you" such that it cannot be changed
        put_rule(self, "baba", "you", positions=(1, height - 2))
        # the agent should be able to break "wall is stop" if needed

        # with this placement, the agent can make "wall is win" and goto "wall"
        # put_rule(self, "wall", "stop", positions=(2, 2))
        # to avoid that we add an unpushable block
        put_rule(self, "wall", "stop", positions=(1, 2))
        # put_obj(self, Wall(), pos=(1, 3))

        if self.color_in_rule:
            # put "obj1 is win" in the corner so that can place a distractor rule block for obj2 such it's impossible to make obj2 win
            put_rule(self, obj1, "win", positions=(width - 4, 1))
        else:
            put_rule(self, obj1[1], "win", positions=(width - 4, 1))

        # Add the distractor rule block
        if self.distractor_rule_block:
            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                place_obj(self, RuleObject(obj3[1]), **self.positions["right_pushable"])
            elif self.distractor_obj:                
                # add distractor rule block for the other object
                # but place it such that it is impossible to make the obj2 win
                positions = self.positions["right_unpushable"]
                pos = positions[np.random.choice(len(positions))]
                put_obj(self, RuleObject(obj2[1]), pos)
            else:
                # don't need to be unpushable because obj2 is not in the env
                place_obj(self, RuleObject(obj2[1]), **self.positions["right_pushable"])

        # Break the rules if needed
        if self.break_stop_rule:
            break_rule(self, ("wall", "stop"), new_pos=self.left_pushable, block_idx=[1, 2])

        if self.break_win_rule:
            win_obj = obj1 if self.color_in_rule else obj1[1]
            block_idx = 1 if self.color_in_rule else 0
            break_rule(self, (win_obj, "win"), new_pos=self.right_pushable, block_idx=block_idx)

        # Place the objects and agent in the rooms
        baba_pos = place_obj(self, "baba", **self.baba_pos)
        obj1_pos = place_obj(self, obj1, **self.obj1_pos)
        if self.distractor_obj:
            place_obj(self, obj2, **self.obj2_pos)

        # add extra distractor win rule
        if self.distractor_win_rule:
            put_rule(self, obj3[1], "win", positions=(width-4, 4))

        # if win rule active and win object on the left, plan = goto win object
        if not self.break_win_rule and obj1_pos[0] < width // 2:
            self.target_plan = f"goto[{obj1[1]}]"
        # if win rule active, win obj on the right and stop rule inactive, plan = goto win object
        elif not self.break_win_rule and obj1_pos[0] > width // 2 and self.break_stop_rule:
            self.target_plan = f"goto[{obj1[1]}]"
        # if win rule active, win obj on the right and stop rule active, plan = break stop rule, goto win object
        elif not self.break_win_rule and obj1_pos[0] > width // 2 and not self.break_stop_rule:
            self.target_plan = f"break[wall is stop], goto[{obj1[1]}]"
        # if win rule inactive and stop rule inactive, plan = make win rule, goto win obj
        elif self.break_win_rule and self.break_stop_rule:
            self.target_plan = f"make[{obj1[1]} is win], goto[{obj1[1]}]"
        # if win rule inactive and stop rule active, plan = break stop rule, make win rule, goto win obj
        elif self.break_win_rule and not self.break_stop_rule:
            self.target_plan = f"break[wall is stop], make[{obj1[1]} is win], goto[{obj1[1]}]"
        else:
            breakpoint()
            raise ValueError("Invalid configuration")


@register("env/two_room-goto_win")
class TwoRoomGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, distractor_rule_block=False, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_obj_rule")
class TwoRoomGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_rule")
class TwoRoomGotoWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_obj")
class TwoRoomGotoWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_rule_block=False, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_obj-irrelevant_rule")
class TwoRoomGotoWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_win_rule")
class TwoRoomGotoWinDistrWinRule(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_win_rule=True, break_stop_rule=True, **kwargs)


# ===== variants of break stop, goto win =====
@register("env/two_room-break_stop-goto_win-distr_obj_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", **kwargs)


@register("env/two_room-break_stop-goto_win-distr_obj")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=False, **kwargs)


@register("env/two_room-break_stop-goto_win-distr_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, **kwargs)

@register("env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/two_room-break_stop-goto_win")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_obj_rule")
class TwoRoomMaybeBreakStopGotoWinDistrObjRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", **kwargs)


@register("env/two_room-maybe_break_stop-goto_win")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_obj")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=True, distractor_rule_block=False, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_rule")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=False, distractor_rule_block=True, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


# ===== variants of make win =====
@register("env/two_room-make_win-distr_obj_rule")
class TwoRoomMakeWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", **kwargs)


@register("env/two_room-make_win-distr_rule")
class TwoRoomMakeWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, **kwargs)


@register("env/two_room-make_win")
class TwoRoomMakeWinNoDistractorEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-make_win-distr_obj-irrelevant_rule")
class TwoRoomMakeWinIrrelevantDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/two_room-make_win-distr_obj")
class TwoRoomMakeWinNoDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_rule_block=False, **kwargs)


@register("env/two_room-make_win-distr_win_rule")
class TwoRoomGotoWinDistrWinRule(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_win_rule=True, break_win_rule=True, break_stop_rule=True, **kwargs)


# ===== variants of break stop, make win =====
@register("env/two_room-break_stop-make_win-distr_obj_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", **kwargs)


@register("env/two_room-break_stop-make_win-distr_rule")
class TwoRoomBreakStopMakeWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, **kwargs)


@register("env/two_room-break_stop-make_win")
class TwoRoomBreakStopMakeWinNoDistractorEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-break_stop-make_win-distr_obj-irrelevant_rule")
class TwoRoomBreakStopMakeWinIrrelevantDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/two_room-break_stop-make_win-distr_obj")
class TwoRoomBreakStopMakeWinNoDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=False, **kwargs)


@register("env/two_room-make_you")
class TwoRoomMakeYouEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=13,
            height=9,
            baba_pos="left_pushable",
            obj1_pos="right_anywhere",
            obj2_pos="right_anywhere",
            break_stop_rule=False,
            break_win_rule=False,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=False,
            color_in_rule=False,
            **kwargs
        ):
        self.color_in_rule = color_in_rule
        self.break_stop_rule = break_stop_rule
        self.break_win_rule = break_win_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor

        pushable_area_size = (width // 2 - 3, height - 6)
        self.positions = {
            # not all left or right space because want it to be possible to push the objects
            "left_pushable": {
                "top": (2, 3),
                "size": pushable_area_size
            },
            "right_pushable": {
                "top": (width // 2 + 2, 3),
                "size": pushable_area_size
            },
            "right_unpushable": [
                # left border
                *[(width-2, 2+i) for i in range(height-4)],
                # bottom border
                *[(width-2-i, height-2) for i in range(width//2-2)],
            ],
            "left_anywhere": {
                "top": (1, 1),
                "size": (width // 2 - 2, height - 2)
            },
            "right_anywhere": {
                # start at y = 2 to prevent having object at the same place as the obj rule block of the win rule for the make_win envs
                "top": (width // 2 + 1, 2),
                "size": (width // 2 - 2, height - 2)
            },
            "anywhere": {
                "top": (1, 2),
                "size": (width - 2, height - 2)
            }
        }
        self.left_pushable = self.positions["left_pushable"]
        self.right_pushable = self.positions["right_pushable"]

        self.baba_pos = self.positions[baba_pos]
        self.obj1_pos = self.positions[obj1_pos]
        self.obj2_pos = self.positions[obj2_pos]

        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add the vertical wall dividing the two rooms
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))

        # Sample the objects
        if self.color_in_rule:
            # obj1 and obj2 can be of the same type but different colors
            obj1_idx, obj2_idx, obj3_idx = np.random.choice(len(OBJECTS), size=3, replace=False)
            obj1 = OBJECTS[obj1_idx]
            obj2 = OBJECTS[obj2_idx]
            # obj3 only useful when distractor_rule_block is True and irrelevant_rule_distractor is True
            obj3 = OBJECTS[obj3_idx]
        else:
            # make sure obj1 and obj2 are different object types
            obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
            obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
            obj1 = (COLORS[obj1_color], NAMES[obj1_name])
            obj2 = (COLORS[obj2_color], NAMES[obj2_name])
            obj3 = (COLORS[obj3_color], NAMES[obj3_name])

        # Add the rules
        # put "baba is you" such that it cannot be changed
        # put_rule(self, "baba", "you", positions=(1, height - 2))
        put_rule(self, "baba", "you", positions=(1, height - 3))
        # the agent should be able to break "wall is stop" if needed

        # with this placement, the agent can make "wall is win" and goto "wall"
        # put_rule(self, "wall", "stop", positions=(2, 2))
        # to avoid that we add an unpushable block
        put_rule(self, "wall", "stop", positions=(1, 1))

        if self.color_in_rule:
            # put "obj1 is win" in the corner so that can place a distractor rule block for obj2 such it's impossible to make obj2 win
            put_rule(self, obj1, "win", positions=(width - 4, 1))
        else:
            put_rule(self, obj1[1], "win", positions=(width - 4, 1))

        # Add the distractor rule block
        if self.distractor_rule_block:
            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                place_obj(self, RuleObject(obj3[1]), **self.positions["left_pushable"])
            elif self.distractor_obj:                
                place_obj(self, RuleObject(obj2[1]), **self.positions["left_pushable"])
            else:
                # don't need to be unpushable because obj2 is not in the env
                place_obj(self, RuleObject(obj2[1]), **self.positions["left_pushable"])

        # Break the rules if needed
        if self.break_stop_rule:
            break_rule(self, ("wall", "stop"), new_pos=self.left_pushable, block_idx=[1, 2])

        if self.break_win_rule:
            win_obj = obj1 if self.color_in_rule else obj1[1]
            block_idx = 1 if self.color_in_rule else 0
            break_rule(self, (win_obj, "win"), new_pos=self.right_pushable, block_idx=block_idx)

        # Place the objects and agent in the rooms
        place_obj(self, "baba", **self.baba_pos)
        place_obj(self, obj1, **self.obj1_pos)
        if self.distractor_obj:
            place_obj(self, obj2, **self.obj2_pos)

        if self.break_win_rule:
            self.target_plan = f"make[{obj2[1]} is you], make[{obj1[1]} is win], goto[{obj1[1]}]"
        else:
            self.target_plan = f"make[{obj2[1]} is you], goto[{obj1[1]}]"


@register("env/two_room-make_you-make_win")
class TwoRoomMakeWinMakeWinEnv(TwoRoomMakeYouEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=True, **kwargs)


@register("env/two_room-make_wall_win")
class TwoRoomMakeWallWinEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=13,
            height=9,
            baba_pos="left_pushable",
            obj1_pos="right_anywhere",
            obj2_pos="right_anywhere",
            break_stop_rule=False,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=True,
            **kwargs
        ):
        self.break_stop_rule = break_stop_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor

        pushable_area_size = (width // 2 - 3, height - 6)
        self.positions = {
            # not all left or right space because want it to be possible to push the objects
            "left_pushable": {
                "top": (2, 3),
                "size": pushable_area_size
            },
            "right_pushable": {
                "top": (width // 2 + 2, 3),
                "size": pushable_area_size
            },
            "right_unpushable": [
                # left border
                *[(width-2, 2+i) for i in range(height-4)],
                # bottom border
                *[(width-2-i, height-2) for i in range(width//2-2)],
            ],
            "left_anywhere": {
                "top": (1, 1),
                "size": (width // 2 - 2, height - 2)
            },
            "right_anywhere": {
                # start at y = 2 to prevent having object at the same place as the obj rule block of the win rule for the make_win envs
                "top": (width // 2 + 1, 2),
                "size": (width // 2 - 2, height - 2)
            },
            "anywhere": {
                "top": (1, 2),
                "size": (width - 2, height - 2)
            }
        }
        self.left_pushable = self.positions["left_pushable"]
        self.right_pushable = self.positions["right_pushable"]

        self.baba_pos = self.positions[baba_pos]
        self.obj1_pos = self.positions[obj1_pos]
        self.obj2_pos = self.positions[obj2_pos]

        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add the vertical wall dividing the two rooms
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))

        # Sample the objects
        obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
        obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
        obj1 = (COLORS[obj1_color], NAMES[obj1_name])
        obj2 = (COLORS[obj2_color], NAMES[obj2_name])
        obj3 = (COLORS[obj3_color], NAMES[obj3_name])

        put_rule(self, "baba", "you", positions=(1, height - 2))
        put_rule(self, "wall", "stop", positions=(2, 2))
        put_rule(self, obj1[1], "win", positions=(width - 4, 1))

        if self.distractor_rule_block:
            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                place_obj(self, RuleObject(obj3[1]), **self.positions["right_pushable"])
            elif self.distractor_obj:
                # add distractor rule block for the other object
                # but place it such that it is impossible to make the obj2 win
                positions = self.positions["right_unpushable"]
                pos = positions[np.random.choice(len(positions))]
                put_obj(self, RuleObject(obj2[1]), pos)
            else:
                # don't need to be unpushable because obj2 is not in the env
                place_obj(self, RuleObject(obj2[1]), **self.positions["right_pushable"])

        # Break the rules if needed
        if self.break_stop_rule:
            break_rule(self, ("wall", "stop"), new_pos=self.left_pushable, block_idx=[1, 2])

        new_pos = self.positions["right_unpushable"][np.random.choice(len(self.positions["right_unpushable"]))]
        break_rule(self, (obj1[1], "win"), new_pos=new_pos, block_idx=0)

        # Place the objects and agent in the rooms
        place_obj(self, "baba", **self.baba_pos)
        place_obj(self, obj1, **self.obj1_pos)
        if self.distractor_obj:
            place_obj(self, obj2, **self.obj2_pos)

        self.target_plan = f"break[wall is stop], make[wall is win], goto[wall]"


@register("env/baba_in_wonderland_baseline_easy")
class BabaInWonderlandBaselineEasyEnv(BabaIsYouEnv):
    SCENARIO_TYPES = (
        "control_static",
        "stop_detour_baba",
        "move_only_wall",
        "shift_enter_key",
        "move_shift_combo",
        "create_move_rule",
        "break_move_rule",
        "create_shift_rule",
        "break_shift_rule",
        "delayed_make_you",
        "lava_hot_basic_detour",
        "defeat_basic_detour",
        "break_hot_for_shortcut",
        "shift_ride_past_lava",
        "create_shift_with_noise",
    )
    ASCII_OBJECTS = CUSTOM_ASCII_OBJECTS or {
        "b": "baba",
        "w": "wall",
        "k": "key",
        "d": "door",
        "o": "crab",
        "l": "lava",
        "m": "flower",
        "h": "brick",
        "q": "hedge",
        "y": "bubble",
        "x": "ice",
        "z": "cog",
        "i": "pipe",
        ";": "robot",
        ",": "bolt",
        ":": "bog",
        "'": "reed",
    }
    ASCII_RULE_OPERATORS = CUSTOM_ASCII_RULE_OPERATORS or {
        "I": "is",
        "+": "and",
    }
    ASCII_RULE_OBJECTS = CUSTOM_ASCII_RULE_OBJECTS or {
        "B": "baba",
        "W": "wall",
        "K": "key",
        "D": "door",
        "O": "crab",
        "L": "lava",
        "{": "flower",
        "}": "brick",
        "=": "hedge",
        ")": "bubble",
        "(": "ice",
        "^": "cog",
        "/": "pipe",
        "&": "robot",
        "%": "bolt",
        "<": "bog",
        ">": "reed",
        "@": "text",
    }
    ASCII_RULE_PROPERTIES = CUSTOM_ASCII_RULE_PROPERTIES or {
        "Y": "you",
        "N": "win",
        "S": "stop",
        "M": "move",
        "H": "shift",
        "[": "sink",
        "!": "open",
        "P": "push",
        "?": "shut",
        "T": "hot",
        "E": "melt",
        "X": "defeat",
        "$": "float",
    }
    SCENARIO_SPECS = {
        "control_static": {
            "layout": (
                "##########",
                "#B.....O.#",
                "#I.....I.#",
                "#Y.....N.#",
                "#........#",
                "#.b......#",
                "#......o.#",
                "##########",
            ),
            "target_plan": "goto[crab]",
        },
        "stop_detour_baba": {
            "layout": (
                "##########",
                "#WIS....D#",
                "#.......I#",
                "#.....w.N#",
                "#B....w.d#",
                "#I....w..#",
                "#Y.b..w..#",
                "##########",
            ),
            "target_plan": "detour around the stop wall and goto[door]",
        },
        "move_only_wall": {
            "layout": (
                "##########",
                "#WIY.DIN.#",
                "#WIM.....#",
                "#........#",
                "#.w...d..#",
                "#........#",
                "#........#",
                "##########",
            ),
            "target_plan": "goto[door] with wall is you + move",
        },
        "shift_enter_key": {
            "layout": (
                "##########",
                "#D......K#",
                "#I......I#",
                "#N.k....Y#",
                "#..wd....#",
                "#....WIH.#",
                "#........#",
                "##########",
            ),
            "target_plan": "enter[shift], goto[door]",
            "dynamic_directions": {
                (3, 4): 0,
            },
        },
        "move_shift_combo": {
            "layout": (
                "##########",
                "#K..WIH..#",
                "#I.......#",
                "#Y.k.wd.D#",
                "#.......I#",
                "#.......N#",
                "#..KIM...#",
                "##########",
            ),
            "target_plan": "goto[door] via move then shift",
            "dynamic_directions": {
                (5, 3): 0,
            },
        },
        "create_move_rule": {
            "layout": (
                "##########",
                "#WI.Mk...#",
                "#KIY..DIN#",
                "#........#",
                "#....w.d.#",
                "#........#",
                "#........#",
                "##########",
            ),
            "target_plan": "make[wall is move], then goto[door]",
            "dynamic_directions": {
                (5, 4): 3,
            },
            "inactive_rules": [
                ("wall", "move", (1, 1), (2, 1), (4, 1)),
            ],
        },
        "break_move_rule": {
            "layout": (
                "##########",
                "#OIY.WWIN#",
                "#...oI...#",
                "#....M...#",
                "#..w.....#",
                "#........#",
                "#........#",
                "##########",
            ),
            "target_plan": "break[wall is move], then goto[wall]",
        },
        "create_shift_rule": {
            "layout": (
                "##########",
                "#WI.Hk...#",
                "#KIY..DIN#",
                "#OIS.....#",
                "#..odwk..#",
                "#........#",
                "#........#",
                "##########",
            ),
            "target_plan": "make[wall is shift], then ride past the stop blocker to the door",
            "dynamic_directions": {
                (5, 4): 2,
            },
            "inactive_rules": [
                ("wall", "shift", (1, 1), (2, 1), (4, 1)),
            ],
        },
        "break_shift_rule": {
            "layout": (
                "##########",
                "#DIYOINW.#",
                "#.....dI.#",
                "#......H.#",
                "#....ow..#",
                "#........#",
                "#........#",
                "##########",
            ),
            "target_plan": "break[wall is shift], then goto[crab]",
            "dynamic_directions": {
                (6, 4): 3,
            },
        },
        "delayed_make_you": {
            "layout": (
                "##########",
                "#DI..YkKO#",
                "#......II#",
                "#......MN#",
                "#.d...o..#",
                "#........#",
                "#........#",
                "##########",
            ),
            "target_plan": "wait until door is you, then goto[crab]",
            "dynamic_directions": {
                (6, 1): 2,
            },
            "inactive_rules": [
                ("door", "you", (1, 1), (2, 1), (5, 1)),
            ],
        },
        "lava_hot_basic_detour": {
            "layout": (
                "##########",
                "#B...L..D#",
                "#I...I..I#",
                "#Y...T..N#",
                "#........#",
                "#.b.lld..#",
                "#BIE.....#",
                "##########",
            ),
            "target_plan": "detour around hot lava, then goto[door]",
        },
        "defeat_basic_detour": {
            "layout": (
                "##########",
                "#B....OIN#",
                "#I.......#",
                "#Y..DIX..#",
                "#........#",
                "#.b.dd.o.#",
                "#........#",
                "##########",
            ),
            "target_plan": "detour around the defeat doors, then goto[crab]",
        },
        "break_hot_for_shortcut": {
            "layout": (
                "##########",
                "#B....L.D#",
                "#I....I.I#",
                "#Y...bT.N#",
                "#.....ld.#",
                "#........#",
                "#BIE.....#",
                "##########",
            ),
            "target_plan": "break[lava is hot], cross the cooled lava, then goto[door]",
        },
        "shift_ride_past_lava": {
            "layout": (
                "##########",
                "#K...W..D#",
                "#I...I..I#",
                "#Y...H..N#",
                "#..kwd...#",
                "#...l....#",
                "#KIE.LIT.#",
                "##########",
            ),
            "target_plan": "ride the shift wall to the win door while avoiding nearby hot lava",
            "dynamic_directions": {
                (4, 4): 0,
            },
        },
        "create_shift_with_noise": {
            "layout": (
                "##########",
                "#KIY.WDIN#",
                "#....IOIM#",
                "#........#",
                "#....H...#",
                "#...kdw..#",
                "#......o.#",
                "##########",
            ),
            "target_plan": "complete wall is shift despite move noise, then use the shift lane",
            "dynamic_directions": {
                (6, 5): 2,
            },
            "inactive_rules": [
                ("wall", "shift", (5, 1), (5, 2), (5, 4)),
            ],
        },
    }

    def __init__(self, width=10, height=8, scenario_type=None, **kwargs):
        self.requested_scenario_type = scenario_type
        self.scenario_type = None
        super().__init__(width=width, height=height, **kwargs)

    def set_agent(self, top=None, size=None, rand_dir=True, max_tries=np.inf):
        pos = None
        for k, e in enumerate(self.grid):
            if e is not None and e.is_agent():
                pos = (k % self.grid.width, k // self.grid.width)
                self.agent_pos = pos
                self.agent_dir = e.dir
                break

        if pos is None:
            pos = (1, self.height - 2)
            self.agent_pos = pos
            self.agent_dir = 0
        return pos

    def _gen_grid(self, width, height, params=None):
        scenario_type = self._select_scenario_type(params=params)
        spec = self.SCENARIO_SPECS.get(scenario_type)
        self._begin_scenario(scenario_type)
        if spec is not None:
            self._build_ascii_layout_from_spec(scenario_type, spec)
            return
        getattr(self, f"_build_{scenario_type}")()

    def _select_scenario_type(self, params=None):
        params = params or {}
        requested = params.get("scenario_type", self.requested_scenario_type)
        if requested is not None:
            if requested not in self.SCENARIO_TYPES:
                raise ValueError(f"Unsupported scenario_type `{requested}`. Expected one of {self.SCENARIO_TYPES}.")
            return requested
        idx = int(np.random.choice(len(self.SCENARIO_TYPES)))
        return self.SCENARIO_TYPES[idx]

    def _begin_scenario(self, scenario_type):
        self.scenario_type = scenario_type
        self.grid = BabaIsYouGrid(self.width, self.height)
        self._build_non_overlapping_border()
        self.objects = {}
        self.active_rules = []
        self.inactive_rule_hints = []
        self.target_plan = ""

    def _build_non_overlapping_border(self):
        for x in range(self.width):
            put_obj(self, Wall(), (x, 0))
            put_obj(self, Wall(), (x, self.height - 1))
        for y in range(1, self.height - 1):
            put_obj(self, Wall(), (0, y))
            put_obj(self, Wall(), (self.width - 1, y))

    def _record_object(self, label, obj, pos):
        self.objects[label] = {
            "type": getattr(obj, "type", None),
            "name": getattr(obj, "name", getattr(obj, "type", None)),
            "position": pos,
            "direction": getattr(obj, "dir", None),
        }

    def _put_dynamic(self, label, obj_name, pos, dir_idx=None):
        obj = make_obj(obj_name)
        if dir_idx is not None:
            obj.dir = dir_idx
        put_obj(self, obj, pos)
        self._record_object(label, obj, pos)
        return obj

    def _put_rule_with_metadata(self, obj, prop, positions, *, active=True):
        put_rule(self, obj, prop, positions=positions)
        if active:
            if isinstance(obj, tuple):
                obj_name = " ".join(obj)
            else:
                obj_name = obj
            self.active_rules.append(f"{obj_name} is {prop}")

    def _put_incomplete_rule(self, obj_name, prop_name, object_pos, is_pos, property_pos):
        put_obj(self, RuleObject(obj_name), object_pos)
        put_obj(self, RuleIs(), is_pos)
        put_obj(self, RuleProperty(prop_name), property_pos)
        self.inactive_rule_hints.append(
            f"{obj_name} is {prop_name} (incomplete: {object_pos}, {is_pos}, {property_pos})"
        )

    def _set_plan(self, plan):
        self.target_plan = plan

    def _build_ascii_layout_from_spec(self, scenario_type, spec):
        layout = spec["layout"]
        if len(layout) != self.height:
            raise ValueError(f"Scenario `{scenario_type}` expected {self.height} rows, got {len(layout)}.")

        dynamic_directions = spec.get("dynamic_directions", {})
        for y, row in enumerate(layout):
            if len(row) != self.width:
                raise ValueError(f"Scenario `{scenario_type}` row {y} expected width {self.width}, got {len(row)}.")
            for x, token in enumerate(row):
                self._place_ascii_token_from_spec(scenario_type, token, x, y, dynamic_directions)
                for stack_index, stack_entry in enumerate(
                    self._get_ascii_stack_entries_from_spec(spec, x, y),
                    start=1,
                ):
                    token_value = str(stack_entry.get("token") or "").strip()
                    if not token_value:
                        continue
                    self._place_ascii_token_from_spec(
                        scenario_type,
                        token_value,
                        x,
                        y,
                        {},
                        dir_idx=stack_entry.get("direction"),
                        label_suffix=f"_stack{stack_index}",
                    )

        self._set_plan(spec["target_plan"])
        for obj_name, prop_name, object_pos, is_pos, property_pos in spec.get("inactive_rules", []):
            self.inactive_rule_hints.append(
                f"{obj_name} is {prop_name} (incomplete: {object_pos}, {is_pos}, {property_pos})"
            )

    def _get_ascii_stack_entries_from_spec(self, spec, x, y):
        raw_stacks = spec.get("cell_stacks", {})
        if not isinstance(raw_stacks, dict):
            return []
        entries = raw_stacks.get((x, y))
        if entries is None:
            entries = raw_stacks.get(f"{x},{y}")
        if not isinstance(entries, list):
            return []
        return entries

    def _place_ascii_token_from_spec(
        self,
        scenario_type,
        token,
        x,
        y,
        dynamic_directions,
        *,
        dir_idx=None,
        label_suffix="",
    ):
        is_border = x in {0, self.width - 1} or y in {0, self.height - 1}
        if token == "#":
            if not is_border:
                raise ValueError(f"Scenario `{scenario_type}` uses `#` inside the playable area at {(x, y)}.")
            return
        if is_border:
            raise ValueError(f"Scenario `{scenario_type}` border must be `#`, found `{token}` at {(x, y)}.")
        if token == ".":
            return
        if token in self.ASCII_OBJECTS:
            label = f"{self.ASCII_OBJECTS[token]}_{x}_{y}{label_suffix}"
            resolved_dir_idx = dynamic_directions.get((x, y)) if dir_idx is None else dir_idx
            self._put_dynamic(label, self.ASCII_OBJECTS[token], (x, y), dir_idx=resolved_dir_idx)
            return
        if token in self.ASCII_RULE_OBJECTS:
            put_obj(self, RuleObject(self.ASCII_RULE_OBJECTS[token]), (x, y))
            return
        if token in self.ASCII_RULE_OPERATORS:
            put_obj(self, make_rule_operator_block(self.ASCII_RULE_OPERATORS[token]), (x, y))
            return
        if token in self.ASCII_RULE_PROPERTIES:
            put_obj(self, RuleProperty(self.ASCII_RULE_PROPERTIES[token]), (x, y))
            return
        raise ValueError(f"Scenario `{scenario_type}` contains unsupported token `{token}` at {(x, y)}.")

    def _build_control_static(self):
        self._put_rule_with_metadata("baba", "you", positions=(1, 1))
        self._put_rule_with_metadata("crab", "win", positions=(5, 1))
        self._put_dynamic("player_baba", "baba", (2, 4))
        self._put_dynamic("goal_crab", "crab", (6, 4))
        self._set_plan("goto[crab]", ["right", "right", "right", "right"])

    def _build_move_only_wall(self):
        self._put_rule_with_metadata("wall", "you", positions=(1, 1))
        self._put_rule_with_metadata("wall", "move", positions=(1, 2))
        self._put_rule_with_metadata("door", "win", positions=(5, 1))
        self._put_dynamic("player_wall", "wall", (2, 4), dir_idx=0)
        self._put_dynamic("goal_door", "door", (6, 4))
        self._set_plan("goto[door] with wall is you + move", ["right", "right"])

    def _build_shift_enter_key(self):
        self._put_rule_with_metadata("key", "you", positions=(1, 1))
        self._put_rule_with_metadata("wall", "shift", positions=(1, 2))
        self._put_rule_with_metadata("door", "win", positions=(5, 1))
        self._put_dynamic("player_key", "key", (3, 3))
        self._put_dynamic("shift_wall", "wall", (3, 4), dir_idx=0)
        self._put_dynamic("goal_door", "door", (4, 4))
        self._set_plan("enter[shift], goto[door]", ["down"])

    def _build_move_shift_combo(self):
        self._put_rule_with_metadata("key", "you", positions=(1, 1))
        self._put_rule_with_metadata("key", "move", positions=(1, 2))
        self._put_rule_with_metadata("wall", "shift", positions=(5, 1))
        self._put_rule_with_metadata("door", "win", positions=(5, 2))
        self._put_dynamic("player_key", "key", (2, 4), dir_idx=0)
        self._put_dynamic("shift_wall", "wall", (4, 4), dir_idx=0)
        self._put_dynamic("goal_door", "door", (5, 4))
        self._set_plan("goto[door] via move then shift", ["right"])

    def _build_create_move_rule(self):
        self._put_rule_with_metadata("key", "you", positions=(1, 2))
        self._put_rule_with_metadata("door", "win", positions=(6, 2))
        self._put_incomplete_rule("wall", "move", (1, 1), (2, 1), (4, 1))
        self._put_dynamic("player_key", "key", (5, 1))
        self._put_dynamic("moving_wall", "wall", (5, 4), dir_idx=3)
        self._put_dynamic("goal_door", "door", (7, 4))
        self._set_plan(
            "make[wall is move], then goto[door]",
            ["left", "down", "down", "down", "right", "right", "right"],
        )

    def _build_break_move_rule(self):
        self._put_rule_with_metadata("crab", "you", positions=(1, 1))
        self._put_rule_with_metadata("wall", "win", positions=(6, 1))
        self._put_rule_with_metadata("wall", "move", positions=[(5, 1), (5, 2), (5, 3)])
        self._put_dynamic("player_crab", "crab", (4, 2))
        self._put_dynamic("goal_wall", "wall", (3, 4), dir_idx=0)
        self._set_plan(
            "break[wall is move], then goto[wall]",
            ["right", "left", "down", "down", "left"],
        )

    def _build_create_shift_rule(self):
        self._put_rule_with_metadata("key", "you", positions=(1, 2))
        self._put_rule_with_metadata("door", "win", positions=(6, 2))
        self._put_incomplete_rule("wall", "shift", (1, 1), (2, 1), (4, 1))
        self._put_dynamic("pusher_key", "key", (5, 1))
        self._put_dynamic("rider_key", "key", (6, 4))
        self._put_dynamic("shift_wall", "wall", (5, 4), dir_idx=2)
        self._put_dynamic("goal_door", "door", (4, 4))
        self._set_plan("make[wall is shift], then ride to door", ["left"])

    def _build_break_shift_rule(self):
        self._put_rule_with_metadata("door", "you", positions=(1, 1))
        self._put_rule_with_metadata("crab", "win", positions=(4, 1))
        self._put_rule_with_metadata("wall", "shift", positions=[(7, 1), (7, 2), (7, 3)])
        self._put_dynamic("player_door", "door", (6, 2))
        self._put_dynamic("shift_wall", "wall", (6, 4), dir_idx=3)
        self._put_dynamic("goal_crab", "crab", (5, 4))
        self._set_plan(
            "break[wall is shift], then goto[crab]",
            ["right", "down", "down", "left", "left"],
        )

    def _build_delayed_make_you(self):
        self._put_rule_with_metadata("key", "move", positions=(6, 2))
        self._put_rule_with_metadata("crab", "win", positions=(6, 3))
        self._put_incomplete_rule("door", "you", (1, 1), (2, 1), (5, 1))
        self._put_dynamic("moving_key", "key", (6, 1), dir_idx=2)
        self._put_dynamic("future_player_door", "door", (2, 4))
        self._put_dynamic("goal_crab", "crab", (6, 4))
        self._set_plan(
            "wait until door is you, then goto[crab]",
            ["idle", "idle", "right", "right", "right", "right"],
        )


# Backward-compatible aliases for earlier naming.
MoveShiftInteractionEasyEnv = BabaInWonderlandBaselineEasyEnv
register("env/move_shift_interaction_easy", obj=BabaInWonderlandBaselineEasyEnv)
register("env/move_shift_interaction", obj=BabaInWonderlandBaselineEasyEnv)


class _BabaCustomAsciiCatalogEnv(BabaInWonderlandBaselineEasyEnv):
    SCENARIO_TYPES = ()
    CUSTOM_DIFFICULTY = "easy"

    def __init__(
        self,
        scenario_type=None,
        allowed_scenario_types=None,
        scenario_split=None,
        split_manifest_path=None,
        **kwargs,
    ):
        self.custom_difficulty = str(self.CUSTOM_DIFFICULTY)
        self.allowed_scenario_types = allowed_scenario_types
        self.scenario_split = scenario_split
        self.split_manifest_path = split_manifest_path
        catalog = load_custom_map_catalog(
            self.custom_difficulty,
            scenario_split=scenario_split,
            allowed_scenario_types=allowed_scenario_types,
            split_manifest_path=split_manifest_path,
        )
        if not catalog:
            root = str(CUSTOM_MAP_ROOT) if CUSTOM_MAP_ROOT is not None else "<custom_maps>"
            raise ValueError(
                f"No custom maps found for difficulty `{self.custom_difficulty}` under "
                f"{root}\\{self.custom_difficulty}."
            )
        self.custom_catalog = catalog
        self.SCENARIO_TYPES = tuple(catalog.keys())
        width, height = self._get_initial_dimensions(requested_scenario_type=scenario_type)
        super().__init__(
            width=width,
            height=height,
            scenario_type=scenario_type,
            **kwargs,
        )

    def _get_initial_dimensions(self, requested_scenario_type=None):
        if custom_map_has_fixed_dimensions(self.custom_difficulty):
            return get_custom_map_dimensions(self.custom_difficulty)

        scenario_name = requested_scenario_type
        if scenario_name not in self.custom_catalog:
            scenario_name = next(iter(self.custom_catalog))
        spec = self.custom_catalog[scenario_name]["spec"]
        return int(spec["width"]), int(spec["height"])

    def _sync_dimensions(self, width, height):
        width = int(width)
        height = int(height)
        if self.width == width and self.height == height:
            return
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.width, self.height, 3 * self.encoding_level),
            dtype="uint8",
        )

    def _gen_grid(self, width, height, params=None):
        scenario_type = self._select_scenario_type(params=params)
        spec = self.custom_catalog[scenario_type]["spec"]
        self._sync_dimensions(spec["width"], spec["height"])
        self._begin_scenario(scenario_type)
        self._build_ascii_layout_from_spec(scenario_type, spec)


@register("env/baba_custom_ascii_easy")
class BabaCustomAsciiEasyEnv(_BabaCustomAsciiCatalogEnv):
    CUSTOM_DIFFICULTY = "easy"


@register("env/baba_custom_ascii_medium")
class BabaCustomAsciiMediumEnv(_BabaCustomAsciiCatalogEnv):
    CUSTOM_DIFFICULTY = "medium"


@register("env/baba_custom_ascii_hard")
class BabaCustomAsciiHardEnv(_BabaCustomAsciiCatalogEnv):
    CUSTOM_DIFFICULTY = "hard"


@register("env/baba_custom_ascii_original")
class BabaCustomAsciiOriginalEnv(_BabaCustomAsciiCatalogEnv):
    CUSTOM_DIFFICULTY = "original"


register("env/baba_custom_ascii", obj=BabaCustomAsciiOriginalEnv)


@register("env/baba_in_wonderland_baseline_medium")
class BabaInWonderlandBaselineMediumEnv(BabaInWonderlandBaselineEasyEnv):
    SCENARIO_TYPES = (
        "editable_shift_goal",
        "move_only_alt_win",
        "wait_for_melted_wall",
        "break_move_with_distractors",
        "delayed_make_you_with_shift_noise",
        "switch_you_for_shift_goal",
        "break_stop_for_door_win",
        "make_key_you_across_stop",
        "make_wall_move_open_path",
        "make_wall_key_then_key_win",
        "break_defeat_gate_for_crab_win",
        "break_hot_lava_gate",
        "shift_lift_win_beside_hot",
        "create_move_melt_wall",
        "switch_key_you_into_hot_pocket",
    )
    SCENARIO_SPECS = {
        "editable_shift_goal": {
            "layout": (
                "############",
                "#.WIH...D..#",
                "#....O..I..#",
                "#....I.....#",
                "#.b..S..w.B#",
                "#.......N.I#",
                "#......d..Y#",
                "#..........#",
                "############",
            ),
            "target_plan": "use the active shift wall to lift N into door is win, then reach the door",
            "dynamic_directions": {
                (8, 4): 3,
            },
        },
        "move_only_alt_win": {
            "layout": (
                "############",
                "#.W.W.OIS..#",
                "#.I.I.OIN..#",
                "#.Y.M......#",
                "#......oo..#",
                "#..wD..IN..#",
                "#..........#",
                "#....d.....#",
                "############",
            ),
            "target_plan": "use wall is move to push D two cells into door is win, then reach the door",
        },
        "wait_for_melted_wall": {
            "layout": (
                "############",
                "#.B.W.W.W..#",
                "#.I.I.I.I..#",
                "#.Y.S.M.E..#",
                "#..........#",
                "#..b.w.ld..#",
                "#..........#",
                "#LIT....DIN#",
                "############",
            ),
            "target_plan": "wait for the moving melt wall to disappear in hot lava, then goto[door]",
        },
        "break_move_with_distractors": {
            "layout": (
                "############",
                "#.O.W...WIN#",
                "#.IoI...DIH#",
                "#.Y.M......#",
                "#..........#",
                "#..........#",
                "#..w.....d.#",
                "#..........#",
                "############",
            ),
            "target_plan": "break wall is move while ignoring an unrelated shift rule",
            "dynamic_directions": {
                (9, 6): 3,
            },
        },
        "delayed_make_you_with_shift_noise": {
            "layout": (
                "############",
                "#......W...#",
                "#DI.Yk.I..o#",
                "#......H...#",
                "#..d.......#",
                "#KIM....OIN#",
                "#...w......#",
                "#..........#",
                "############",
            ),
            "target_plan": "wait for key is move to complete door is you, then reach the crab while ignoring unrelated wall is shift noise",
            "dynamic_directions": {
                (5, 2): 2,
            },
            "inactive_rules": [
                ("door", "you", (1, 2), (2, 2), (4, 2)),
            ],
        },
        "switch_you_for_shift_goal": {
            "layout": (
                "############",
                "#B......D..#",
                "#I......I..#",
                "#Y......N..#",
                "#......o...#",
                "#.b...okwd.#",
                "#.K.IYoWIH.#",
                "#OIS...o...#",
                "############",
            ),
            "target_plan": "switch control from baba to key, then use the shift wall to carry the remote key into the win door",
            "inactive_rules": [
                ("key", "you", (2, 6), (4, 6), (5, 6)),
            ],
        },
        "break_stop_for_door_win": {
            "layout": (
                "############",
                "#D..W...B..#",
                "#I.bI...I..#",
                "#N..S...Y..#",
                "#wwwwwwwwww#",
                "#.........d#",
                "#..........#",
                "#..........#",
                "############",
            ),
            "target_plan": "break[wall is stop], then goto[door]",
        },
        "make_key_you_across_stop": {
            "layout": (
                "############",
                "#BIY.OISDIN#",
                "#KI.Y.o....#",
                "#.....o....#",
                "#.b...o.kd.#",
                "#.....o....#",
                "#.....o....#",
                "#.....o....#",
                "############",
            ),
            "target_plan": "make[key is you] across the stop barrier, then use the remote key to reach the door",
        },
        "make_wall_move_open_path": {
            "layout": (
                "############",
                "#B.......D.#",
                "#I..WI.MbI.#",
                "#Y.......N.#",
                "#ooooowoooo#",
                "#..........#",
                "#.......d..#",
                "#OIS.......#",
                "############",
            ),
            "target_plan": "make[wall is move], let the moving wall open the stop barrier gap, then cross to the door",
            "dynamic_directions": {
                (6, 3): 1,
            },
            "inactive_rules": [
                ("wall", "move", (4, 1), (5, 1), (7, 1)),
            ],
        },
        "make_wall_key_then_key_win": {
            "layout": (
                "############",
                "#BIY.KIN...#",
                "#OIS..o....#",
                "#WI.K.o....#",
                "#.b..wo.k..#",
                "#.....o....#",
                "#.....o....#",
                "#.....o....#",
                "############",
            ),
            "target_plan": "make[wall is key], then reach the transformed key win on the near side",
            "inactive_rules": [
                ("wall", "key", (1, 3), (2, 3), (4, 3)),
            ],
        },
        "break_defeat_gate_for_crab_win": {
            "layout": (
                "############",
                "#......o.O.#",
                "#........I.#",
                "#........N.#",
                "#dddddddddd#",
                "#.....D....#",
                "#.....I....#",
                "#BIYb.X....#",
                "############",
            ),
            "target_plan": "break[door is defeat], then cross the former defeat gate and goto[crab]",
        },
        "break_hot_lava_gate": {
            "layout": (
                "############",
                "#......dD..#",
                "#.......I..#",
                "#.......N..#",
                "#llllllllll#",
                "#B.......B.#",
                "#I..LIT..I.#",
                "#Y.b.....E.#",
                "############",
            ),
            "target_plan": "break[lava is hot], then cross the lava gate and goto[door]",
        },
        "shift_lift_win_beside_hot": {
            "layout": (
                "############",
                "#WIH....D.L#",
                "#.......I.I#",
                "#B......w.T#",
                "#I......w.l#",
                "#E.....Nw.l#",
                "#..........#",
                "#..BIYb..d.#",
                "############",
            ),
            "target_plan": "push N onto the active shift column, let it rise into door is win, then goto[door]",
            "dynamic_directions": {
                (8, 3): 3,
                (8, 4): 3,
                (8, 5): 3,
            },
        },
        "create_move_melt_wall": {
            "layout": (
                "############",
                "#B.O...D.LB#",
                "#I.I...IdII#",
                "#Y.S...N.TE#",
                "#oooowloooo#",
                "#WW........#",
                "#II........#",
                "#ES...WI.Mb#",
                "############",
            ),
            "target_plan": "make[wall is move], let the stop+melt wall walk into the hot lava and open the lane, then goto[door]",
            "dynamic_directions": {
                (5, 4): 0,
            },
            "inactive_rules": [
                ("wall", "move", (6, 7), (7, 7), (9, 7)),
            ],
        },
        "switch_key_you_into_hot_pocket": {
            "layout": (
                "############",
                "#D....L...B#",
                "#I....I.dlI#",
                "#N....Tll.E#",
                "#.....l....#",
                "#B..kll....#",
                "#I.........#",
                "#Y.bK.IY...#",
                "############",
            ),
            "target_plan": "make[key is you], then send the non-melting key through the hot pocket to the door",
            "inactive_rules": [
                ("key", "you", (4, 7), (6, 7), (7, 7)),
            ],
        },
    }

    def __init__(self, width=12, height=9, scenario_type=None, **kwargs):
        super().__init__(width=width, height=height, scenario_type=scenario_type, **kwargs)

    def _build_editable_shift_goal(self):
        self._put_rule_with_metadata("baba", "you", positions=[(2, 1), (2, 2), (2, 3)])
        self._put_rule_with_metadata("wall", "shift", positions=[(4, 1), (4, 2), (4, 3)])
        self._put_rule_with_metadata("door", "win", positions=(7, 1))
        self._put_rule_with_metadata("key", "move", positions=(7, 2))
        self._put_dynamic("player_baba", "baba", (4, 6))
        self._put_dynamic("shift_wall", "wall", (5, 6), dir_idx=0)
        self._put_dynamic("goal_door", "door", (6, 6))
        self._put_dynamic("distractor_key", "key", (9, 6), dir_idx=2)
        self._set_plan(
            "editable you/shift rules with irrelevant key is move noise",
            ["right"],
        )

    def _build_move_only_alt_win(self):
        self._put_rule_with_metadata("wall", "you", positions=[(2, 1), (2, 2), (2, 3)])
        self._put_rule_with_metadata("wall", "move", positions=[(4, 1), (4, 2), (4, 3)])
        self._put_rule_with_metadata("crab", "win", positions=(7, 1))
        self._put_rule_with_metadata("key", "push", positions=(7, 2))
        self._put_incomplete_rule("door", "win", (7, 3), (8, 3), (10, 3))
        self._put_dynamic("player_wall", "wall", (3, 6), dir_idx=0)
        self._put_dynamic("goal_crab", "crab", (7, 6))
        self._put_dynamic("alt_goal_door", "door", (9, 6))
        self._put_dynamic("distractor_key", "key", (10, 6))
        self._set_plan(
            "move-only wall path with editable win targets and distractor rules",
            ["right", "right"],
        )

    def _build_create_shift_with_noise(self):
        self._put_rule_with_metadata("key", "you", positions=[(2, 1), (2, 2), (2, 3)])
        self._put_rule_with_metadata("door", "win", positions=(8, 1))
        self._put_rule_with_metadata("crab", "move", positions=(8, 2))
        self._put_incomplete_rule("wall", "shift", (5, 1), (5, 2), (5, 4))
        self._put_dynamic("player_key", "key", (5, 5))
        self._put_dynamic("shift_wall", "wall", (7, 5), dir_idx=2)
        self._put_dynamic("goal_door", "door", (6, 5))
        self._put_dynamic("noise_crab", "crab", (10, 6), dir_idx=0)
        self._set_plan(
            "complete wall is shift despite move noise, then use the shift lane",
            ["up", "right", "right", "down"],
        )

    def _build_break_move_with_distractors(self):
        self._put_rule_with_metadata("crab", "you", positions=[(2, 1), (2, 2), (2, 3)])
        self._put_rule_with_metadata("wall", "move", positions=[(4, 1), (4, 2), (4, 3)])
        self._put_rule_with_metadata("wall", "win", positions=(8, 1))
        self._put_rule_with_metadata("door", "shift", positions=(8, 2))
        self._put_dynamic("player_crab", "crab", (3, 2))
        self._put_dynamic("goal_wall", "wall", (3, 6), dir_idx=0)
        self._put_dynamic("distractor_door", "door", (9, 6), dir_idx=3)
        self._set_plan(
            "break wall is move while ignoring an unrelated shift rule",
            ["right", "left", "down", "down", "down", "down"],
        )

    def _build_delayed_make_you_with_shift_noise(self):
        self._put_rule_with_metadata("wall", "shift", positions=(8, 1))
        self._put_rule_with_metadata("door", "push", positions=(8, 2))
        self._put_rule_with_metadata("key", "move", positions=(8, 3))
        self._put_rule_with_metadata("crab", "win", positions=(8, 4))
        self._put_incomplete_rule("door", "you", (2, 1), (3, 1), (5, 1))
        self._put_dynamic("moving_key", "key", (6, 1), dir_idx=2)
        self._put_dynamic("future_player_door", "door", (3, 6))
        self._put_dynamic("goal_crab", "crab", (7, 6))
        self._put_dynamic("noise_shift_wall", "wall", (10, 6), dir_idx=3)
        self._set_plan(
            "wait for door is you while unrelated shift/push rules stay editable",
            ["idle", "right", "right", "right", "right"],
        )

    def _build_switch_you_for_shift_goal(self):
        self._put_rule_with_metadata("baba", "you", positions=[(2, 1), (2, 2), (2, 3)])
        self._put_rule_with_metadata("door", "win", positions=(8, 1))
        self._put_rule_with_metadata("crab", "stop", positions=(8, 2))
        self._put_rule_with_metadata("wall", "shift", positions=[(4, 1), (4, 2), (4, 3)])
        self._put_incomplete_rule("key", "you", (6, 3), (7, 3), (9, 3))
        self._put_dynamic("player_baba", "baba", (10, 3))
        self._put_dynamic("future_player_key", "key", (4, 6))
        self._put_dynamic("shift_wall", "wall", (5, 6), dir_idx=0)
        self._put_dynamic("goal_door", "door", (6, 6))
        self._put_dynamic("distractor_crab", "crab", (10, 6))
        self._set_plan(
            "switch control from baba to key, then use shift to reach the win door",
            ["left", "right"],
        )


MoveShiftInteractionMediumEnv = BabaInWonderlandBaselineMediumEnv
register("env/move_shift_interaction_medium", obj=BabaInWonderlandBaselineMediumEnv)


@register("env/baba_in_wonderland_baseline_hard")
class BabaInWonderlandBaselineHardEnv(BabaInWonderlandBaselineEasyEnv):
    SCENARIO_TYPES = (
        "make_key_you_then_break_hot_gate",
        "move_make_door_you_on_lava_island",
        "boxed_start_key_then_break_hot_pocket",
        "shared_melt_crab_move_barrier",
        "moving_lava_crossfire_escape",
        "switch_key_you_split_rooms",
        "door_you_door_win_under_stop_row",
        "wall_to_key_then_key_you_ring",
        "crab_to_door_then_door_you_l_maze",
        "move_make_wall_you_then_door_win",
        "delayed_make_door_you_by_move",
        "center_biy_cross_swap",
        "move_shift_remote_door_win",
        "shift_make_key_you_then_door_win",
        "triple_chain_from_other_you",
    )

    ASCII_OBJECTS = {
        "b": "baba",
        "w": "wall",
        "k": "key",
        "d": "door",
        "o": "crab",
        "l": "lava",
        "m": "flower",
        "h": "brick",
        "q": "hedge",
        "y": "bubble",
        "x": "ice",
        "z": "cog",
        "i": "pipe",
        ";": "robot",
        ",": "bolt",
        ":": "bog",
        "'": "reed",
    }
    ASCII_RULE_OPERATORS = CUSTOM_ASCII_RULE_OPERATORS or {
        "I": "is",
        "+": "and",
    }
    ASCII_RULE_OBJECTS = {
        "B": "baba",
        "W": "wall",
        "K": "key",
        "D": "door",
        "O": "crab",
        "L": "lava",
        "{": "flower",
        "}": "brick",
        "=": "hedge",
        ")": "bubble",
        "(": "ice",
        "^": "cog",
        "/": "pipe",
        "&": "robot",
        "%": "bolt",
        "<": "bog",
        ">": "reed",
        "@": "text",
    }
    ASCII_RULE_PROPERTIES = {
        "Y": "you",
        "N": "win",
        "S": "stop",
        "M": "move",
        "H": "shift",
        "[": "sink",
        "!": "open",
        "T": "hot",
        "E": "melt",
        "X": "defeat",
        "?": "shut",
        "$": "float",
    }
    HARD_SCENARIOS = {
        "make_key_you_then_break_hot_gate": {
            "layout": (
                "###############",
                "#DIN..B...L.K.#",
                "#d..l.I...I.I.#",
                "#...l.Yk..T.E.#",
                "#oooooo.......#",
                "#..K.IY.......#",
                "#OIS..b.......#",
                "###############",
            ),
            "target_plan": "make[key is you], break[lava is hot] from the upper chamber, then use the cooled lava gate to reach the win door",
        },
        "move_make_door_you_on_lava_island": {
            "layout": (
                "###############",
                "#OIN..llll.B..#",
                "#o....l..l.I..#",
                "#lllll.d.l.E..#",
                "#lllllLITl....#",
                "#..DIE.KIM....#",
                "#b.....DI.Yk..#",
                "###############",
            ),
            "target_plan": "let key is move auto-complete door is you, then use the isolated door on the lava island to break lava is hot and cross the cooled sea to the win crab",
            "dynamic_directions": {
                (11, 6): 2,
            },
        },
        "boxed_start_key_then_break_hot_pocket": {
            "layout": (
                "###############",
                "#B......ODIN.K#",
                "#Iooooo.I...dI#",
                "#YoKI.o.S...LE#",
                "#.o..Yo..kldI.#",
                "#.o.b.o.....T.#",
                "#.ooooo.......#",
                "###############",
            ),
            "target_plan": "from inside the boxed stop pen, complete key is you, then use the outside melt-key to break lava is hot and cross the cooled pocket to the win door",
        },
        "shared_melt_crab_move_barrier": {
            "layout": (
                "###############",
                "#BIY....DIN..B#",
                "#b...OI.M....I#",
                "#..........OIE#",
                "#oooooolllllll#",
                "#.d.....LIT...#",
                "#OIS..........#",
                "###############",
            ),
            "target_plan": "complete crab is move, let the shared melt crabs slide into the hot lava half and open a gap in the mixed barrier, then drop to the win door",
            "dynamic_directions": {
                (1, 4): 2,
                (2, 4): 2,
                (3, 4): 2,
                (4, 4): 2,
                (5, 4): 2,
                (6, 4): 2,
            },
        },
        "moving_lava_crossfire_escape": {
            "layout": (
                "###############",
                "#B......D.LIMd#",
                "#I..O...I.I...#",
                "#Y..I...N.T...#",
                "#..oS.l..o..l.#",
                "#.o...........#",
                "#b..o....BIE..#",
                "###############",
            ),
            "target_plan": "dodge the horizontal and vertical moving lava patrols, keep the melting baba alive, and reach the win door in the upper right",
            "dynamic_directions": {
                (6, 4): 0,
                (12, 4): 3,
            },
        },
        "switch_key_you_split_rooms": {
            "layout": (
                "###############",
                "#BIY..WIS.DIN.#",
                "#b....w....d..#",
                "#.....w...k...#",
                "#.....w.......#",
                "#.....w.......#",
                "#..K.IY.......#",
                "###############",
            ),
            "target_plan": "use the left room to complete key is you, then control the separated key across the split rooms to reach the win door",
        },
        "door_you_door_win_under_stop_row": {
            "layout": (
                "###############",
                "#.....DIN.....#",
                "#.....d.......#",
                "#ooooooooooooo#",
                "#OIS.....D.IY.#",
                "#BIY....b.....#",
                "#.............#",
                "###############",
            ),
            "target_plan": "break through the stop row by completing door is you, then control the upper door that is already win",
        },
        "wall_to_key_then_key_you_ring": {
            "layout": (
                "###############",
                "#BIY....DIN...#",
                "#...OIS.......#",
                "#..ooooo......#",
                "#..o.wdo......#",
                "#..ooooo......#",
                "#W.IK..K.IY.b.#",
                "###############",
            ),
            "target_plan": "complete wall is key and then key is you, so the transformed ring object can reach the win door",
        },
        "crab_to_door_then_door_you_l_maze": {
            "layout": (
                "###############",
                "#KIY....W.IY..#",
                "#OIS.ooooooo..#",
                "#.k..oDI.N.o..#",
                "#....o.d.w.o..#",
                "#....ooooooo..#",
                "#.............#",
                "###############",
            ),
            "target_plan": "make[wall is you], make[door is win], then use the ring wall to reach the door",
        },
        "move_make_wall_you_then_door_win": {
            "layout": (
                "###############",
                "#.......o...D.#",
                "#kW..IY.o.w.Id#",
                "#.......o.....#",
                "#...OIS.o...N.#",
                "#...BIY.o...w.#",
                "#KIM....o.....#",
                "###############",
            ),
            "target_plan": "let key is move auto-complete wall is you, then use the right-side wall to push N upward into door is win and reach the door",
        },
        "delayed_make_door_you_by_move": {
            "layout": (
                "###############",
                "#KIM....WIS...#",
                "#kD.IY........#",
                "#..OI.........#",
                "#....N........#",
                "#....d........#",
                "#...w.www.o...#",
                "###############",
            ),
            "target_plan": "wait for key is move to complete door is you, then make[crab is win] and reach the crab",
        },
        "center_biy_cross_swap": {
            "layout": (
                "###############",
                "#..WIS.w.kD.IN#",
                "#..b.B.w......#",
                "#....I.wwwwww.#",
                "#.K.IY.w......#",
                "#......w..d...#",
                "#......w......#",
                "###############",
            ),
            "target_plan": "break the centered baba is you into key is you, then make[door is win] on the right side and reach the door",
        },
        "move_shift_remote_door_win": {
            "layout": (
                "###############",
                "#B.KI..o......#",
                "#I...M.o......#",
                "#Y.WI..o......#",
                "#.b..Hdo..owNk#",
                "#OIS...o.DI...#",
                "#......o......#",
                "###############",
            ),
            "target_plan": "make[wall is shift], then make[key is move] so the remote key pushes N onto the downward shift wall to complete door is win, then reach the left-side door",
            "dynamic_directions": {
                (11, 4): 1,
                (13, 4): 2,
            },
        },
        "shift_make_key_you_then_door_win": {
            "layout": (
                "###############",
                "#KI.Y.o.WI.H..#",
                "#.b...o..k..D.#",
                "#B....o..N..I.#",
                "#I....o.owww..#",
                "#Y....o....d..#",
                "#OIS..o.......#",
                "###############",
            ),
            "target_plan": "make[key is you], then make[wall is shift], push N down onto the shift lane so it slides into vertical door is win, and reach the door",
        },
        "triple_chain_from_other_you": {
            "layout": (
                "###############",
                "#KIY..BIS.O.IN#",
                "#..bbb........#",
                "#..bwb.W.IK...#",
                "#..bob.K.I.D..#",
                "#..bbb.D.IY...#",
                "#..........k..#",
                "###############",
            ),
            "target_plan": "start from key control, make[wall is key], make[key is door], make[door is you], then make[crab is win] and use the enclosed door to reach the crab",
        },
    }

    def __init__(self, width=15, height=8, scenario_type=None, **kwargs):
        super().__init__(width=width, height=height, scenario_type=scenario_type, **kwargs)

    def _gen_grid(self, width, height, params=None):
        scenario_type = self._select_scenario_type(params=params)
        self._begin_scenario(scenario_type)
        self._build_hard_ascii_layout(scenario_type)

    def _build_hard_ascii_layout(self, scenario_type):
        spec = self.HARD_SCENARIOS[scenario_type]
        layout = spec["layout"]
        if len(layout) != self.height:
            raise ValueError(f"Scenario `{scenario_type}` expected {self.height} rows, got {len(layout)}.")

        for y, row in enumerate(layout):
            if len(row) != self.width:
                raise ValueError(f"Scenario `{scenario_type}` row {y} expected width {self.width}, got {len(row)}.")
            for x, token in enumerate(row):
                self._place_ascii_token(scenario_type, token, x, y)

        self._set_plan(spec["target_plan"])

    def _place_ascii_token(self, scenario_type, token, x, y):
        spec = self.HARD_SCENARIOS[scenario_type]
        dynamic_directions = spec.get("dynamic_directions", {})
        is_border = x in {0, self.width - 1} or y in {0, self.height - 1}
        if token == "#":
            if not is_border:
                raise ValueError(f"Scenario `{scenario_type}` uses `#` inside the playable area at {(x, y)}.")
            return
        if is_border:
            raise ValueError(f"Scenario `{scenario_type}` border must be `#`, found `{token}` at {(x, y)}.")
        if token == ".":
            return
        if token in self.ASCII_OBJECTS:
            label = f"{self.ASCII_OBJECTS[token]}_{x}_{y}"
            dir_idx = dynamic_directions.get((x, y))
            self._put_dynamic(label, self.ASCII_OBJECTS[token], (x, y), dir_idx=dir_idx)
            return
        if token in self.ASCII_RULE_OBJECTS:
            put_obj(self, RuleObject(self.ASCII_RULE_OBJECTS[token]), (x, y))
            return
        if token in self.ASCII_RULE_OPERATORS:
            put_obj(self, make_rule_operator_block(self.ASCII_RULE_OPERATORS[token]), (x, y))
            return
        if token in self.ASCII_RULE_PROPERTIES:
            put_obj(self, RuleProperty(self.ASCII_RULE_PROPERTIES[token]), (x, y))
            return
        raise ValueError(f"Scenario `{scenario_type}` contains unsupported token `{token}` at {(x, y)}.")


MoveShiftInteractionHardEnv = BabaInWonderlandBaselineHardEnv
register("env/move_shift_interaction_hard", obj=BabaInWonderlandBaselineHardEnv)

