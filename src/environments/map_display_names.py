from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from src.environments.custom_map_spec import list_custom_map_paths, load_custom_map_spec


_DISPLAY_NAME_SCAN_ORDER = ("original", "easy", "medium", "hard")
_cached_display_name_index: Optional[Dict[str, str]] = None


def _iter_lookup_keys(*values: Any) -> Iterable[str]:
    seen: set[str] = set()
    for raw_value in values:
        text = str(raw_value or "").strip()
        if not text:
            continue
        for candidate in (text, Path(text).name):
            normalized = str(candidate or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            yield normalized
            if normalized.lower().endswith(".json"):
                stem = normalized[:-5].strip()
                if stem and stem not in seen:
                    seen.add(stem)
                    yield stem


def _first_non_empty_candidate(*values: Any) -> Optional[str]:
    for raw_value in values:
        text = str(raw_value or "").strip()
        if text:
            return text
    return None


def _list_custom_map_index_paths() -> Tuple[Path, ...]:
    return tuple(
        path.resolve()
        for difficulty in _DISPLAY_NAME_SCAN_ORDER
        for path in list_custom_map_paths(difficulty)
    )


def _build_custom_map_display_name_index(paths: Iterable[Path]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in paths:
        try:
            spec = load_custom_map_spec(path)
        except (FileNotFoundError, ValueError, OSError):
            continue
        display_name = str(spec.get("display_name") or "").strip()
        if not display_name:
            continue
        for key in _iter_lookup_keys(
            spec.get("scenario_name"),
            path.stem,
            path.name,
        ):
            index.setdefault(key, display_name)
    return index


def build_custom_map_display_name_index() -> Dict[str, str]:
    return _build_custom_map_display_name_index(_list_custom_map_index_paths())


def _cached_or_built_display_name_index() -> Dict[str, str]:
    global _cached_display_name_index
    if _cached_display_name_index is None:
        _cached_display_name_index = build_custom_map_display_name_index()
    return _cached_display_name_index


def refresh_custom_map_display_name_index() -> Dict[str, str]:
    global _cached_display_name_index
    _cached_display_name_index = build_custom_map_display_name_index()
    return dict(_cached_display_name_index)


def get_custom_map_display_name_index() -> Dict[str, str]:
    return dict(_cached_or_built_display_name_index())


def resolve_custom_map_spec_display_name(
    spec: Mapping[str, Any],
    path: str | Path,
    *,
    index: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    display_name = str(spec.get("display_name") or "").strip()
    if display_name:
        return display_name
    resolved_path = Path(path)
    return resolve_custom_map_display_name(
        spec.get("scenario_name"),
        resolved_path.stem,
        resolved_path.name,
        index=index,
    )


def resolve_custom_map_display_name(
    *candidates: Any,
    index: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    lookup = index if index is not None else _cached_or_built_display_name_index()
    for candidate in candidates:
        for key in _iter_lookup_keys(candidate):
            display_name = str(lookup.get(key) or "").strip()
            if display_name:
                return display_name
    return None


def resolve_map_display_name(
    *candidates: Any,
    index: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    resolved = resolve_custom_map_display_name(*candidates, index=index)
    if resolved:
        return resolved
    return _first_non_empty_candidate(*candidates)
