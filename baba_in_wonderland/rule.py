from collections import defaultdict
from typing import Iterable, List, Optional, Sequence, Tuple


def _add_rule(rule, ruleset):
    if '_rule_' not in ruleset:
        ruleset['_rule_'] = []
    ruleset['_rule_'].append(rule)

    if 'property' in rule and 'object' in rule:
        ruleset[rule['property']][rule['object']] = True

        if 'obj_color' in rule:
            color = rule['obj_color']
            color_key = rule['object'] + "_color"

            colors = ruleset[rule['property']].get(color_key, set())
            colors.add(color)
            ruleset[rule['property']][color_key] = colors

    elif 'object1' in rule and 'object2' in rule:
        replace_list = ruleset.get('replace', [])
        replace_list.append((rule['object1'], rule['object2']))
        ruleset['replace'] = replace_list


def _finalize_replace_rules(ruleset):
    replace_list = list(ruleset.get('replace', []))
    if len(replace_list) == 0:
        return

    # In Baba Is You, `X IS X` prevents X from transforming into any object.
    blocked_sources = {source for source, target in replace_list if source == target}
    if len(blocked_sources) == 0:
        return

    ruleset['replace'] = [
        (source, target)
        for source, target in replace_list
        if source not in blocked_sources
    ]


def _parse_conjoined_items(block_list, allowed_types: Sequence[str]):
    if not block_list:
        return None

    trimmed = list(block_list)
    while trimmed and trimmed[0].type == 'rule_and':
        trimmed = trimmed[1:]
    while trimmed and trimmed[-1].type == 'rule_and':
        trimmed = trimmed[:-1]
    if not trimmed:
        return None

    items = []
    expect_item = True
    allowed = set(allowed_types)

    for block in trimmed:
        if block is None:
            return None
        if expect_item:
            if block.type not in allowed:
                return None
            items.append(block)
        else:
            if block.type != 'rule_and':
                return None
        expect_item = not expect_item

    if expect_item:
        return None
    return items


def extract_rules(block_list):
    """
    Take a left/right rule segment around an IS block and expand it into one or more
    rule dicts. Supports conjunctions like `BABA AND WALL IS YOU AND PUSH`.
    """
    if len(block_list) < 3:
        return []
    for block in block_list:
        if block is None:
            return []

    if len(block_list) == 3:
        left_block, middle_block, right_block = block_list
        if middle_block.type != 'rule_is':
            return []
        if left_block.type == "rule_object" and right_block.type == "rule_property":
            return [{'object': left_block.object, 'property': right_block.property}]
        if left_block.type == "rule_object" and right_block.type == "rule_object":
            return [{'object1': left_block.object, 'object2': right_block.object}]
        return []

    is_index = None
    for idx, block in enumerate(block_list):
        if block.type == 'rule_is':
            is_index = idx
            break
    if is_index is None:
        return []

    left_blocks = list(block_list[:is_index])
    right_blocks = list(block_list[is_index + 1:])
    if not left_blocks or not right_blocks:
        return []

    obj_color = None
    if left_blocks and left_blocks[0].type == 'rule_color':
        obj_color = left_blocks[0].obj_color
        left_blocks = left_blocks[1:]

    subjects = _parse_conjoined_items(left_blocks, ('rule_object',))
    if not subjects:
        return []

    predicates = _parse_conjoined_items(right_blocks, ('rule_property', 'rule_object'))
    if not predicates:
        return []

    rules = []
    for subject in subjects:
        for predicate in predicates:
            if predicate.type == 'rule_property':
                rule = {
                    'object': subject.object,
                    'property': predicate.property,
                }
                if obj_color is not None:
                    rule['obj_color'] = obj_color
                rules.append(rule)
                continue

            if obj_color is not None:
                continue
            rules.append({
                'object1': subject.object,
                'object2': predicate.object,
            })
    return rules


def extract_rule(block_list):
    """
    Backward-compatible helper for callers that expect a single rule.
    """
    rules = extract_rules(block_list)
    if len(rules) != 1:
        return None
    return rules[0]


def maybe_add_rule(block_list, ruleset):
    """
    If the blocks form one or more valid rules, add them to the ruleset.
    Args:
        block_list: list of blocks
        ruleset: dict with the active rules
    """
    rules = extract_rules(block_list)
    if len(rules) == 0:
        return False

    for rule in rules:
        _add_rule(rule, ruleset)
    return True


def inside_grid(grid, pos):
    """
    Return true if pos is inside the boundaries of the grid
    """
    i, j = pos
    inside_grid = (i >= 0 and i < grid.width) and (j >= 0 and j < grid.height)
    return inside_grid


def _collect_subject_blocks(grid, start_pos: Tuple[int, int], delta: Tuple[int, int]) -> List:
    i, j = start_pos
    dx, dy = delta
    blocks = []
    seen_object = False
    expect_object = True

    while inside_grid(grid, (i, j)):
        block = grid.get(i, j)
        if block is None:
            break
        if expect_object:
            if block.type == 'rule_and' and not seen_object:
                blocks.append(block)
            elif block.type == 'rule_object':
                blocks.append(block)
                seen_object = True
                expect_object = False
            elif block.type == 'rule_color' and seen_object:
                blocks.append(block)
                break
            else:
                break
        else:
            if block.type != 'rule_and':
                break
            blocks.append(block)
            expect_object = True
        i += dx
        j += dy

    blocks.reverse()
    return blocks


def _collect_predicate_blocks(grid, start_pos: Tuple[int, int], delta: Tuple[int, int]) -> List:
    i, j = start_pos
    dx, dy = delta
    blocks = []
    seen_item = False
    expect_item = True

    while inside_grid(grid, (i, j)):
        block = grid.get(i, j)
        if block is None:
            break
        if expect_item:
            if block.type == 'rule_and' and not seen_item:
                blocks.append(block)
            elif block.type in {'rule_object', 'rule_property'}:
                blocks.append(block)
                seen_item = True
                expect_item = False
            else:
                break
        else:
            if block.type != 'rule_and':
                break
            blocks.append(block)
            expect_item = True
        i += dx
        j += dy

    return blocks


def extract_ruleset(grid, default_ruleset=None):
    """
    Construct the ruleset from the grid. Called every time a RuleBlock is pushed.
    """
    ruleset = defaultdict(dict)
    ruleset.update(default_ruleset) if default_ruleset is not None else None

    if not isinstance(grid, Iterable):
        grid = grid.grid

    for k, e in enumerate(grid):
        if e is None or e.type != 'rule_is':
            continue

        i, j = k % grid.width, k // grid.width
        assert k == j * grid.width + i

        horizontal = (
            _collect_subject_blocks(grid, (i - 1, j), (-1, 0))
            + [e]
            + _collect_predicate_blocks(grid, (i + 1, j), (1, 0))
        )
        maybe_add_rule(horizontal, ruleset)

        vertical = (
            _collect_subject_blocks(grid, (i, j - 1), (0, -1))
            + [e]
            + _collect_predicate_blocks(grid, (i, j + 1), (0, 1))
        )
        maybe_add_rule(vertical, ruleset)

    _finalize_replace_rules(ruleset)
    return ruleset
