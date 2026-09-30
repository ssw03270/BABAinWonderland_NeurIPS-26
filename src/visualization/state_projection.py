from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


_STATE_TYPE_ALIASES = {
    "a": "rule_noun",
    "b": "rule_operator",
    "c": "rule_property",
    "d": "world_object",
    "rule_noun": "rule_noun",
    "rule_operator": "rule_operator",
    "rule_property": "rule_property",
    "world_object": "world_object",
    "noun": "rule_noun",
    "operator": "rule_operator",
    "property": "rule_property",
    "object": "world_object",
}


def _normalize_word(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower()
    return normalized or "unknown"


def normalize_visual_object_type(raw_type: Any) -> str:
    return _STATE_TYPE_ALIASES.get(_normalize_word(raw_type), _normalize_word(raw_type))


def build_visualization_config(
    word_aliases: Optional[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    reverse_aliases: Dict[str, Dict[str, str]] = {}
    collisions: Dict[str, set[str]] = {}
    if isinstance(word_aliases, Mapping):
        for raw_type, raw_mapping in word_aliases.items():
            obj_type = normalize_visual_object_type(raw_type)
            if not obj_type or not isinstance(raw_mapping, Mapping):
                continue
            type_reverse = reverse_aliases.setdefault(obj_type, {})
            type_collisions = collisions.setdefault(obj_type, set())
            for raw_word, raw_alias in raw_mapping.items():
                if not isinstance(raw_word, str) or not isinstance(raw_alias, str):
                    continue
                word = _normalize_word(raw_word)
                alias = _normalize_word(raw_alias)
                if not word or not alias:
                    continue
                existing = type_reverse.get(alias)
                if existing is None:
                    type_reverse[alias] = word
                    continue
                if existing != word:
                    type_collisions.add(alias)
        for obj_type, aliased_words in collisions.items():
            for alias in aliased_words:
                reverse_aliases.get(obj_type, {}).pop(alias, None)
    return {
        "reverseWordAliases": {
            str(obj_type): dict(sorted(alias_map.items()))
            for obj_type, alias_map in sorted(reverse_aliases.items())
            if alias_map
        }
    }


def _resolve_reverse_aliases(
    visual_config: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, str]]:
    if not isinstance(visual_config, Mapping):
        return {}
    raw_reverse = visual_config.get("reverseWordAliases")
    if not isinstance(raw_reverse, Mapping):
        raw_reverse = visual_config.get("reverse_word_aliases")
    if not isinstance(raw_reverse, Mapping):
        return {}

    normalized: Dict[str, Dict[str, str]] = {}
    for raw_type, raw_mapping in raw_reverse.items():
        obj_type = normalize_visual_object_type(raw_type)
        if not obj_type or not isinstance(raw_mapping, Mapping):
            continue
        alias_map: Dict[str, str] = {}
        for raw_alias, raw_word in raw_mapping.items():
            if not isinstance(raw_alias, str) or not isinstance(raw_word, str):
                continue
            alias = _normalize_word(raw_alias)
            word = _normalize_word(raw_word)
            if not alias or not word:
                continue
            alias_map[alias] = word
        if alias_map:
            normalized[obj_type] = alias_map
    return normalized


def canonicalize_visual_word(
    obj_type: Any,
    word: Any,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> str:
    normalized_type = normalize_visual_object_type(obj_type)
    normalized_word = _normalize_word(word)
    reverse_aliases = _resolve_reverse_aliases(visual_config)
    type_aliases = reverse_aliases.get(normalized_type, {})
    return type_aliases.get(normalized_word, normalized_word)


def canonicalize_state_for_visualization(
    state: Mapping[str, Any],
    visual_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(state, Mapping):
        return {}

    normalized_state = dict(state)
    raw_objects = state.get("objects")
    if not isinstance(raw_objects, list):
        return normalized_state

    objects = []
    for raw_obj in raw_objects:
        if not isinstance(raw_obj, Mapping):
            continue
        obj = dict(raw_obj)
        obj_type = normalize_visual_object_type(obj.get("type"))
        word = obj.get("word", obj.get("text"))
        obj["type"] = obj_type
        obj["word"] = canonicalize_visual_word(
            obj_type=obj_type,
            word=word,
            visual_config=visual_config,
        )
        obj.pop("text", None)
        obj.pop("sprite_key", None)
        obj.pop("spriteKey", None)
        objects.append(obj)
    normalized_state["objects"] = objects
    return normalized_state
