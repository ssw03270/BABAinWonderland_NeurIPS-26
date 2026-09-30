import math

import numpy as np

from baba_in_wonderland.utils import add_img_text
from baba_in_wonderland.rendering import fill_coords, point_in_circle, point_in_rect, point_in_triangle, rotate_fn

properties = [
    'is_stop',
    'is_push',
    'is_goal',
    'is_defeat',
    'is_sink',
    'is_agent',
    'is_pull',
    'is_move',
    'is_shift',
    'is_hot',
    'is_melt',
    'is_open',
    'is_shut',
    'is_float',
]

objects = [
    'fwall',
    'fflag',
    'fdoor',
    "fkey",
    "flava",
    'baba',
    'falgae',
    'fcrab',
    'fgrass',
    'fjelly',
    'fkeke',
    'flove',
    'fpillar',
    'frock',
    'fskull',
    'fstar',
    'ftile',
    'fwater',
    'fflower',
    'fbrick',
    'fhedge',
    'fbubble',
    'fice',
    'fcog',
    'fpipe',
    'frobot',
    'fbolt',
    'fbog',
    'freed',
]

name_mapping = {
    'fwall': 'wall',
    'fflag': 'flag',
    'fdoor': 'door',
    'fkey': 'key',
    'falgae': 'algae',
    'fcrab': 'crab',
    'fgrass': 'grass',
    'fjelly': 'jelly',
    'fkeke': 'keke',
    'flove': 'love',
    'fpillar': 'pillar',
    'frock': 'rock',
    'fskull': 'skull',
    'fstar': 'star',
    'ftile': 'tile',
    'fwater': 'water',
    'fflower': 'flower',
    'fbrick': 'brick',
    'fhedge': 'hedge',
    'fbubble': 'bubble',
    'fice': 'ice',
    'fcog': 'cog',
    'fpipe': 'pipe',
    'frobot': 'robot',
    'fbolt': 'bolt',
    'fbog': 'bog',
    'freed': 'reed',
    'is_push': 'push',
    'is_stop': 'stop',
    'is_goal': 'win',
    'is_defeat': 'defeat',
    'is_sink': 'sink',
    'is': 'is',
    'is_agent': 'you',
    'is_pull': 'pull',
    'is_move': 'move',
    'is_shift': 'shift',
    'is_hot': 'hot',
    'is_melt': 'melt',
    'is_open': 'open',
    'is_shut': 'shut',
    'is_float': 'float',
    'text': 'text',
    'and': 'and',
    'flava': 'lava',
}
# by default, add the displayed name is the type of the object
name_mapping.update({o: o for o in objects if o not in name_mapping})
name_mapping_inverted = {v: k for k, v in name_mapping.items()}


def normalize_rule_name(value: str) -> str:
    return str(value).strip().lower()


def is_text_rule_name(value: str) -> bool:
    return normalize_rule_name(value) == "text"

TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5, "white": 6}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "crab": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

def add_object_types(object_types):
    last_idx = len(OBJECT_TO_IDX)-1
    OBJECT_TO_IDX.update({
        t: last_idx+1+i
    for i, t in enumerate(object_types)})


def add_color_types(color_types):
    last_idx = len(COLOR_TO_IDX)-1
    COLOR_TO_IDX.update({
        t: last_idx+1+i
    for i, t in enumerate(color_types)})


add_color_types(name_mapping.values())
add_object_types(objects)
add_object_types(['rule', 'rule_object', 'rule_is', 'rule_and', 'rule_property', 'rule_color'])


def make_obj(name: str, color: str = None):
    """
    Make an object from a string name
    """
    kwargs = {'color': color} if color is not None else {}

    # TODO: make it more general
    if name == "fwall" or name == "wall":
        obj_cls = FWall
    elif name == "fflag" or name == "flag":
        obj_cls = FFlag
    elif name == "fkey" or name == "key":
        obj_cls = FKey
    elif name == "flava" or name == "lava":
        obj_cls = FLava
    elif name == "fdoor" or name == "door":
        obj_cls = FDoor
    elif name == "baba":
        obj_cls = Baba
    elif name == "falgae" or name == "algae":
        obj_cls = FAlgae
    elif name == "fcrab" or name == "crab":
        obj_cls = FCrab
    elif name == "fgrass" or name == "grass":
        obj_cls = FGrass
    elif name == "fjelly" or name == "jelly":
        obj_cls = FJelly
    elif name == "fkeke" or name == "keke":
        obj_cls = FKeke
    elif name == "flove" or name == "love":
        obj_cls = FLove
    elif name == "fpillar" or name == "pillar":
        obj_cls = FPillar
    elif name == "frock" or name == "rock":
        obj_cls = FRock
    elif name == "fskull" or name == "skull":
        obj_cls = FSkull
    elif name == "fstar" or name == "star":
        obj_cls = FStar
    elif name == "ftile" or name == "tile":
        obj_cls = FTile
    elif name == "fwater" or name == "water":
        obj_cls = FWater
    elif name == "fflower" or name == "flower":
        obj_cls = FFlower
    elif name == "fbrick" or name == "brick":
        obj_cls = FBrick
    elif name == "fhedge" or name == "hedge":
        obj_cls = FHedge
    elif name == "fbubble" or name == "bubble":
        obj_cls = FBubble
    elif name == "fice" or name == "ice":
        obj_cls = FIce
    elif name == "fcog" or name == "cog":
        obj_cls = FCog
    elif name == "fpipe" or name == "pipe":
        obj_cls = FPipe
    elif name == "frobot" or name == "robot":
        obj_cls = FRobot
    elif name == "fbolt" or name == "bolt":
        obj_cls = FBolt
    elif name == "fbog" or name == "bog":
        obj_cls = FBog
    elif name == "freed" or name == "reed":
        obj_cls = FReed
    else:
        raise ValueError(name)

    return obj_cls(**kwargs)


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self._type_color_key = type + "_color"
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def is_agent(self):
        return False

    def is_move(self):
        return False

    def is_shift(self):
        return False

    def is_hot(self):
        return False

    def is_melt(self):
        return False

    def is_goal(self):
        return False

    def is_defeat(self):
        return False

    def is_sink(self):
        return False

    def is_float(self):
        return False

    def is_open(self):
        return False

    def is_shut(self):
        return False

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def is_push(self):
        """Can the agent push this?"""
        return False

    def is_pull(self):
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def set_ruleset(self, ruleset):
        self._ruleset = ruleset

    def get_ruleset(self):
        return getattr(self, "_ruleset", None)

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "crab":
            v = make_obj("crab", color=color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        else:
            try:
                v = make_obj(obj_type, color=color)
            except ValueError as exc:
                raise AssertionError("unknown object type in decode '%s'" % obj_type) from exc

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Wall(WorldObj):
    def __init__(self, color="grey"):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class RuleBlock(WorldObj):
    """
    By default, rule blocks can be pushed by the agent.
    """
    def __init__(self, name, type, color, is_push=True):
        super().__init__(type, color)
        self._is_push = is_push
        self.name = name = name_mapping.get(name, name)
        self.dir = 0
        self.margin = 10
        img = np.zeros((96-2*self.margin, 96-2*self.margin, 3), np.uint8)
        add_img_text(img, name)
        self.img = img

    def set_ruleset(self, ruleset):
        self._ruleset = ruleset

    def get_ruleset(self):
        return getattr(self, "_ruleset", None)

    def _get_text_rule_prop(self, prop):
        ruleset = getattr(self, "_ruleset", None)
        if ruleset is None:
            return False
        ruleset_dict = getattr(ruleset, "ruleset_dict", ruleset)
        text_rule_name = "text"
        text_color_name = "text_color"

        if prop == 'is_stop':
            stop_rules = ruleset_dict['is_stop']
            pull_rules = ruleset_dict['is_pull']
            pull_colors = pull_rules.get(text_color_name)
            if pull_rules.get(text_rule_name, False) and (
                not pull_colors or self.color in pull_colors
            ):
                stop_rules[text_rule_name] = True
            agent_rules = ruleset_dict['is_agent']
            agent_colors = agent_rules.get(text_color_name)
            if agent_rules.get(text_rule_name, False) and (
                not agent_colors or self.color in agent_colors
            ):
                stop_rules[text_rule_name] = True

        prop_rules = ruleset_dict[prop]
        if prop_rules.get(text_rule_name, False):
            color_set = prop_rules.get(text_color_name)
            if (not color_set) or (self.color in color_set):
                return True
        return False

    def can_overlap(self):
        return (not self.is_stop()) and (not self.is_push())

    def is_push(self):
        return self._is_push or self._get_text_rule_prop("is_push")

    def is_pull(self):
        return self._get_text_rule_prop("is_pull")

    def is_agent(self):
        return self._get_text_rule_prop("is_agent")

    def is_move(self):
        return self._get_text_rule_prop("is_move")

    def is_shift(self):
        return self._get_text_rule_prop("is_shift")

    def is_hot(self):
        return self._get_text_rule_prop("is_hot")

    def is_melt(self):
        return self._get_text_rule_prop("is_melt")

    def is_goal(self):
        return self._get_text_rule_prop("is_goal")

    def is_defeat(self):
        return self._get_text_rule_prop("is_defeat")

    def is_sink(self):
        return self._get_text_rule_prop("is_sink")

    def is_float(self):
        return self._get_text_rule_prop("is_float")

    def is_stop(self):
        return self._get_text_rule_prop("is_stop")

    def is_open(self):
        return self._get_text_rule_prop("is_open")

    def is_shut(self):
        return self._get_text_rule_prop("is_shut")

    def render(self, img):
        fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), [235, 235, 235])
        img[self.margin:-self.margin, self.margin:-self.margin] = self.img

    # TODO: different encodings of the rule blocks for the agent observation
    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # RuleBlock characterized by their name instead of color
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.name], 0)


class RuleObject(RuleBlock):
    def __init__(self, obj, is_push=True):
        obj = name_mapping_inverted[obj] if obj not in objects else obj
        # TODO: red push is win (push is a rule_obj but not in objects)
        # assert obj in objects, "{} not in {}".format(obj, objects)

        super().__init__(obj, 'rule_object', 'purple', is_push=is_push)
        self.object = obj


class RuleProperty(RuleBlock):
    def __init__(self, property, is_push=True):
        property = name_mapping_inverted[property] if property not in properties else property
        assert property in properties, "{} not in {}".format(property, properties)

        super().__init__(property, 'rule_property', 'purple', is_push=is_push)
        self.property = property


class RuleIs(RuleBlock):
    def __init__(self, is_push=True):
        super().__init__('is', 'rule_is', 'purple', is_push=is_push)


class RuleAnd(RuleBlock):
    def __init__(self, is_push=True):
        super().__init__('and', 'rule_and', 'purple', is_push=is_push)


class RuleColor(RuleBlock):
    def __init__(self, obj_color, is_push=True):
        assert obj_color in COLOR_TO_IDX, "{} not in {}".format(obj_color, COLOR_TO_IDX)

        super().__init__(obj_color, 'rule_color', 'purple', is_push=is_push)
        self.obj_color = obj_color


class Ruleset:
    """
    Each object in the env has a reference to the ruleset object, which is automatically updated (would have to manually
    update it if were using a dict instead).
    """
    def __init__(self, ruleset_dict):
        self.ruleset_dict = ruleset_dict
        self.version = 0

    def set(self, ruleset_dict):
        self.ruleset_dict = ruleset_dict
        self.version += 1

    def __getitem__(self, item):
        return self.ruleset_dict[item]

    def __setitem__(self, key, value):
        self.ruleset_dict[key] = value
        self.version += 1

    def bump(self):
        self.version += 1

    def __str__(self):
        return f'Ruleset dict: {self.ruleset_dict}'

    # TODO: cause infinite loop when using vec env
    # def __getattr__(self, item):
    #     return getattr(self.ruleset_dict, item)

    def get(self, *args, **kwargs):
        return self.ruleset_dict.get(*args, **kwargs)


def make_text_rule_block_for_object(obj_name: str, is_push: bool = True) -> RuleObject:
    normalized = normalize_rule_name(obj_name)
    display_name = name_mapping.get(normalized, normalized)
    return RuleObject(display_name, is_push=is_push)



def make_prop_fn(prop: str):
    """
    Make a method that retrieves the property of an instance of FlexibleWorldObj in the ruleset
    """
    prop_name = prop
    is_stop_rule = prop_name == "is_stop"

    def get_prop(self: FlexibleWorldObj):
        # retrieve the type and color specific to the instance 'self' (the function is the same for all instances)
        ruleset = getattr(self, "_ruleset", None)
        if ruleset is None:
            return False
        ruleset_dict = getattr(ruleset, "ruleset_dict", ruleset)
        typ = self.type
        color = self.color
        typ_color_name = getattr(self, "_type_color_key", typ + "_color")

        # TODO: cleaner way to implement implicit rules? e.g. is_pull, is_agent implies is_stop
        if is_stop_rule:
            stop_rules = ruleset_dict["is_stop"]
            pull_rules = ruleset_dict["is_pull"]
            if pull_rules.get(typ, False):
                pull_colors = pull_rules.get(typ_color_name)
                if pull_colors and color in pull_colors:
                    stop_rules[typ] = True
            agent_rules = ruleset_dict["is_agent"]
            if agent_rules.get(typ, False):
                agent_colors = agent_rules.get(typ_color_name)
                if agent_colors and color in agent_colors:
                    stop_rules[typ] = True

        # check for rules specific to a color e.g. {"is_goal": {"fcrab": True, "fcrab_color": [0]}}
        prop_rules = ruleset_dict[prop_name]
        if prop_rules.get(typ, False):  # object type set to True
            # if no specified color or object fits color specifications
            color_set = prop_rules.get(typ_color_name)
            if (not color_set) or (color in color_set):
                return True
        return False

    return get_prop


class FlexibleWorldObj(WorldObj):
    def __init__(self, type, color):
        assert type in objects, "{} not in {}".format(type, objects)
        super().__init__(type, color)
        self.name = name_mapping[type]  # pretty name
        # direction in which the object is facing
        self.dir = 0  # order: right, down, left, up

        for prop in properties:
            # create a method for each property and bind it to the class (same for all the instances of that class)
            setattr(self.__class__, prop, make_prop_fn(prop))

    def set_ruleset(self, ruleset):
        self._ruleset = ruleset

    def get_ruleset(self):
        return self._ruleset

    # compatibility with WorldObj
    def can_overlap(self):
        return (not self.is_stop()) and (not self.is_push())


class LabeledFlexibleWorldObj(FlexibleWorldObj):
    LABEL = "OBJ"

    def render(self, img):
        fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), COLORS[self.color])
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (20, 20, 20))
        add_img_text(img, self.LABEL)


class FWall(FlexibleWorldObj):
    def __init__(self, color="grey"):
        super().__init__("fwall", color)

    def render(self, img):
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), COLORS[self.color])


class FFlag(LabeledFlexibleWorldObj):
    LABEL = "FLG"

    def __init__(self, color="yellow"):
        super().__init__("fflag", color)


class FDoor(FlexibleWorldObj):
    def __init__(self, color="red"):
        super().__init__("fdoor", color)

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # TODO: don't need to encode the state
        state = 0
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

        # Draw door handle
        fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class FKey(FlexibleWorldObj):
    def __init__(self, color="blue"):
        super().__init__("fkey", color)

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class FLava(FlexibleWorldObj):
    def __init__(self, color="red"):
        super().__init__("flava", color)

    def render(self, img):
        fill_coords(img, point_in_rect(0.05, 0.95, 0.35, 0.95), (180, 40, 0))
        fill_coords(img, point_in_triangle((0.10, 0.95), (0.28, 0.40), (0.42, 0.95)), (255, 140, 0))
        fill_coords(img, point_in_triangle((0.32, 0.95), (0.50, 0.28), (0.68, 0.95)), (255, 190, 0))
        fill_coords(img, point_in_triangle((0.58, 0.95), (0.76, 0.42), (0.90, 0.95)), (255, 140, 0))


class Baba(FlexibleWorldObj):
    def __init__(self, color="white"):
        super().__init__("baba", color)

    def render(self, img):
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, (255, 255, 255))


class FAlgae(LabeledFlexibleWorldObj):
    LABEL = "ALG"

    def __init__(self, color="green"):
        super().__init__("falgae", color)


class FCrab(LabeledFlexibleWorldObj):
    LABEL = "CRB"

    def __init__(self, color="red"):
        super().__init__("fcrab", color)


class FGrass(LabeledFlexibleWorldObj):
    LABEL = "GRS"

    def __init__(self, color="green"):
        super().__init__("fgrass", color)


class FJelly(LabeledFlexibleWorldObj):
    LABEL = "JLY"

    def __init__(self, color="purple"):
        super().__init__("fjelly", color)


class FKeke(LabeledFlexibleWorldObj):
    LABEL = "KEK"

    def __init__(self, color="yellow"):
        super().__init__("fkeke", color)


class FLove(LabeledFlexibleWorldObj):
    LABEL = "LUV"

    def __init__(self, color="red"):
        super().__init__("flove", color)


class FPillar(LabeledFlexibleWorldObj):
    LABEL = "PIL"

    def __init__(self, color="grey"):
        super().__init__("fpillar", color)


class FRock(LabeledFlexibleWorldObj):
    LABEL = "RCK"

    def __init__(self, color="grey"):
        super().__init__("frock", color)


class FSkull(LabeledFlexibleWorldObj):
    LABEL = "SKL"

    def __init__(self, color="white"):
        super().__init__("fskull", color)


class FStar(LabeledFlexibleWorldObj):
    LABEL = "STR"

    def __init__(self, color="yellow"):
        super().__init__("fstar", color)


class FTile(LabeledFlexibleWorldObj):
    LABEL = "TIL"

    def __init__(self, color="grey"):
        super().__init__("ftile", color)


class FWater(LabeledFlexibleWorldObj):
    LABEL = "WTR"

    def __init__(self, color="blue"):
        super().__init__("fwater", color)


class FFlower(LabeledFlexibleWorldObj):
    LABEL = "FLR"

    def __init__(self, color="purple"):
        super().__init__("fflower", color)


class FBrick(LabeledFlexibleWorldObj):
    LABEL = "BRK"

    def __init__(self, color="red"):
        super().__init__("fbrick", color)


class FHedge(LabeledFlexibleWorldObj):
    LABEL = "HDG"

    def __init__(self, color="green"):
        super().__init__("fhedge", color)


class FBubble(LabeledFlexibleWorldObj):
    LABEL = "BUB"

    def __init__(self, color="blue"):
        super().__init__("fbubble", color)


class FIce(LabeledFlexibleWorldObj):
    LABEL = "ICE"

    def __init__(self, color="blue"):
        super().__init__("fice", color)


class FCog(LabeledFlexibleWorldObj):
    LABEL = "COG"

    def __init__(self, color="grey"):
        super().__init__("fcog", color)


class FPipe(LabeledFlexibleWorldObj):
    LABEL = "PIP"

    def __init__(self, color="grey"):
        super().__init__("fpipe", color)


class FRobot(LabeledFlexibleWorldObj):
    LABEL = "BOT"

    def __init__(self, color="yellow"):
        super().__init__("frobot", color)


class FBolt(LabeledFlexibleWorldObj):
    LABEL = "BLT"

    def __init__(self, color="yellow"):
        super().__init__("fbolt", color)


class FBog(LabeledFlexibleWorldObj):
    LABEL = "BOG"

    def __init__(self, color="green"):
        super().__init__("fbog", color)


class FReed(LabeledFlexibleWorldObj):
    LABEL = "RED"

    def __init__(self, color="green"):
        super().__init__("freed", color)
