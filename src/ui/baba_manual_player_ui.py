"""Sprite-driven Baba-style renderer used by the manual player."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageSequence

from .unified_scene import (
    BoardScene,
    SceneCell,
    SceneObject,
    build_custom_map_board_scene,
    build_state_board_scene,
)
from src.visualization import canonicalize_state_for_visualization


RESAMPLE_NEAREST = Image.Resampling.NEAREST


RULE_OBJECT_TYPES = {"rule_noun", "rule_operator", "rule_property"}
RULE_END_TYPES = {"rule_noun", "rule_property"}
HAZARD_PROPERTIES = {"defeat", "hot", "sink", "melt"}
FALLBACK_WORLD_COLORS = {
    "door": ("#D8A16B", "#F3DEC6"),
    "brick": ("#A36E62", "#E7D3CF"),
    "key": ("#D9C36B", "#F4ECC1"),
    "box": ("#A593C7", "#E0D7F0"),
    "baba": ("#F0EBE0", "#FFFDF8"),
    "wall": ("#767676", "#CACACA"),
    "algae": ("#5E8D69", "#D8EAD8"),
    "bog": ("#5E7E56", "#D7E4D2"),
    "bolt": ("#D7BF63", "#F8EDC1"),
    "crab": ("#CF6F65", "#F5D6D2"),
    "grass": ("#7A9A75", "#D9E6D5"),
    "jelly": ("#A789C8", "#EADFEE"),
    "keke": ("#D8C46C", "#F5ECC0"),
    "love": ("#D47A95", "#F5D6E1"),
    "pillar": ("#8C867A", "#E1DCCF"),
    "pipe": ("#9C8C78", "#E6DCCF"),
    "reed": ("#7E9B5C", "#E0ECCD"),
    "rock": ("#8D8D8D", "#D6D6D6"),
    "robot": ("#D1B56C", "#F5EAC3"),
    "skull": ("#D8D8D8", "#FBFBFB"),
    "star": ("#D8BC63", "#FAEDBD"),
    "tile": ("#B3A28B", "#EEE5D8"),
    "water": ("#6C99C8", "#DAE8F6"),
    "lava": ("#C97A6B", "#F0D2CA"),
    "flag": ("#C97A6B", "#F0D2CA"),
    "flower": ("#C98A9A", "#F0D8DE"),
    "hedge": ("#6F9A63", "#D8EACF"),
    "bubble": ("#7FB7D9", "#DDF2FB"),
    "cog": ("#9E9589", "#E6DED2"),
    "ice": ("#8FB8DA", "#E4F2FB"),
}
RULE_TILE_COLORS = {
    "rule_noun": ("#080808", "#C67968", "#F0D8D2"),
    "rule_operator": ("#080808", "#C8C2B4", "#F4F0E7"),
    "rule_property": ("#080808", "#968E7D", "#EEE8DD"),
}
PANEL_FILL = "#000000"
PANEL_OUTLINE = "#D8D2C4"
CARD_FILL = "#050505"
CARD_OUTLINE = "#B7B0A1"
BACKGROUND_TOP = "#000000"
HEADER_ACCENT = "#D07C6D"
TEXT_PRIMARY = "#F3EFE6"
TEXT_MUTED = "#A5A095"
GRID_GLOW = "#D7CFBF"
STATUS_COLORS = {
    "running": ("#0A0A0A", "#7A9381"),
    "win": ("#0C0B07", "#D3C5A1"),
    "terminated": ("#0F0908", "#B87A6F"),
    "truncated": ("#11100A", "#A69469"),
    "save": ("#090B0E", "#7D93A1"),
}


def normalize_word(word: Any) -> str:
    if not isinstance(word, str):
        return "unknown"
    value = word.strip().lower()
    return value or "unknown"


def extract_grid_size(state: Dict[str, Any]) -> Tuple[int, int]:
    raw = state.get("grid_size")
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 2
        and isinstance(raw[0], int)
        and isinstance(raw[1], int)
    ):
        return int(raw[0]), int(raw[1])

    max_x = -1
    max_y = -1
    for obj in state.get("objects", []):
        if not isinstance(obj, dict):
            continue
        pos = obj.get("position")
        if (
            isinstance(pos, (list, tuple))
            and len(pos) == 2
            and isinstance(pos[0], int)
            and isinstance(pos[1], int)
        ):
            max_x = max(max_x, int(pos[0]))
            max_y = max(max_y, int(pos[1]))
    return max_x + 1, max_y + 1


def extract_rule_triples(state: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    by_pos: Dict[Tuple[int, int], List[Tuple[str, str]]] = defaultdict(list)
    for obj in state.get("objects", []):
        if not isinstance(obj, dict):
            continue
        obj_type = normalize_word(obj.get("type"))
        if obj_type not in RULE_OBJECT_TYPES:
            continue
        pos = obj.get("position")
        if not (
            isinstance(pos, (list, tuple))
            and len(pos) == 2
            and isinstance(pos[0], int)
            and isinstance(pos[1], int)
        ):
            continue
        word = normalize_word(obj.get("word") or obj.get("text"))
        by_pos[(int(pos[0]), int(pos[1]))].append((obj_type, word))

    def collect_subject_segment(
        start: Tuple[int, int],
        delta: Tuple[int, int],
    ) -> List[Tuple[str, str]]:
        x, y = int(start[0]), int(start[1])
        dx, dy = int(delta[0]), int(delta[1])
        tokens: List[Tuple[str, str]] = []
        seen_noun = False
        expect_noun = True
        while True:
            options = by_pos.get((x, y), [])
            token = next(
                (
                    (obj_type, word)
                    for obj_type, word in options
                    if obj_type in {"rule_noun", "rule_property", "rule_operator"}
                ),
                None,
            )
            if token is None:
                break
            obj_type, word = token
            if expect_noun:
                if obj_type == "rule_operator" and word == "and" and not seen_noun:
                    tokens.append(token)
                elif obj_type == "rule_noun":
                    tokens.append(token)
                    seen_noun = True
                    expect_noun = False
                else:
                    break
            else:
                if obj_type != "rule_operator" or word != "and":
                    break
                tokens.append(token)
                expect_noun = True
            x += dx
            y += dy
        tokens.reverse()
        return tokens

    def collect_predicate_segment(
        start: Tuple[int, int],
        delta: Tuple[int, int],
    ) -> List[Tuple[str, str]]:
        x, y = int(start[0]), int(start[1])
        dx, dy = int(delta[0]), int(delta[1])
        tokens: List[Tuple[str, str]] = []
        seen_item = False
        expect_item = True
        while True:
            options = by_pos.get((x, y), [])
            token = next(
                (
                    (obj_type, word)
                    for obj_type, word in options
                    if obj_type in {"rule_noun", "rule_property", "rule_operator"}
                ),
                None,
            )
            if token is None:
                break
            obj_type, word = token
            if expect_item:
                if obj_type == "rule_operator" and word == "and" and not seen_item:
                    tokens.append(token)
                elif obj_type in {"rule_noun", "rule_property"}:
                    tokens.append(token)
                    seen_item = True
                    expect_item = False
                else:
                    break
            else:
                if obj_type != "rule_operator" or word != "and":
                    break
                tokens.append(token)
                expect_item = True
            x += dx
            y += dy
        return tokens

    def parse_conjoined_words(
        tokens: Sequence[Tuple[str, str]],
        allowed_item_types: Sequence[str],
    ) -> Optional[Tuple[str, ...]]:
        if len(tokens) == 0:
            return None
        trimmed = list(tokens)
        while trimmed and trimmed[0] == ("rule_operator", "and"):
            trimmed = trimmed[1:]
        while trimmed and trimmed[-1] == ("rule_operator", "and"):
            trimmed = trimmed[:-1]
        if len(trimmed) == 0:
            return None
        allowed = {normalize_word(item) for item in allowed_item_types}
        words: List[str] = []
        expect_item = True
        for obj_type, word in trimmed:
            if expect_item:
                if obj_type not in allowed:
                    return None
                words.append(word)
            else:
                if obj_type != "rule_operator" or word != "and":
                    return None
            expect_item = not expect_item
        if expect_item:
            return None
        return tuple(words)

    triples: List[Tuple[str, str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for (x, y), operators in by_pos.items():
        if not any(obj_type == "rule_operator" and word == "is" for obj_type, word in operators):
            continue
        for dx, dy in ((1, 0), (0, 1)):
            subjects = parse_conjoined_words(
                collect_subject_segment((x - dx, y - dy), (-dx, -dy)),
                ("rule_noun",),
            )
            if not subjects:
                continue
            predicate_tokens = collect_predicate_segment((x + dx, y + dy), (dx, dy))
            if len(predicate_tokens) == 0:
                continue
            predicates = parse_conjoined_words(predicate_tokens, RULE_END_TYPES)
            if not predicates:
                continue
            for subject in subjects:
                for predicate in predicates:
                    triple = (subject, "is", predicate)
                    if triple not in seen:
                        seen.add(triple)
                        triples.append(triple)
    return triples


def extract_rule_sentences(state: Dict[str, Any]) -> List[str]:
    return [f"{lhs.upper()} IS {rhs.upper()}" for lhs, _, rhs in extract_rule_triples(state)]


def extract_active_property_map(state: Dict[str, Any]) -> Dict[str, set[str]]:
    active: Dict[str, set[str]] = defaultdict(set)
    for lhs, _, rhs in extract_rule_triples(state):
        active[lhs].add(rhs)
    return active


def resolve_status_key(*, terminated: bool, truncated: bool, reward: float) -> str:
    if truncated:
        return "truncated"
    if terminated:
        if float(reward) > 0:
            return "win"
        return "terminated"
    return "running"


def determine_highlight_kind(properties: set[str]) -> Optional[str]:
    if "you" in properties:
        return "you"
    if "win" in properties:
        return "win"
    if properties.intersection(HAZARD_PROPERTIES):
        return "danger"
    return None


@dataclass(frozen=True)
class ActionLogEntry:
    step_index: int
    action_name: str
    reward: float
    terminated: bool = False
    truncated: bool = False
    note: str = ""
    event_type: str = "action"

    @property
    def label(self) -> str:
        if self.event_type == "reset":
            return "RESET"
        if self.event_type == "save":
            return "SAVE"
        if self.event_type == "system":
            return "INFO"
        return str(self.action_name or "idle").upper()

    @property
    def status_key(self) -> str:
        if self.event_type == "save":
            return "save"
        return resolve_status_key(
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            reward=float(self.reward),
        )

    @property
    def reward_text(self) -> str:
        return f"{float(self.reward):+0.3f}"


@dataclass(frozen=True)
class SpriteAnimation:
    frames: Tuple[Image.Image, ...]
    durations_ms: Tuple[int, ...]
    total_duration_ms: int

    def resolve_frame_index(self, elapsed_ms: int) -> int:
        if len(self.frames) <= 1:
            return 0
        if self.total_duration_ms <= 0:
            return int(elapsed_ms) % len(self.frames)

        target = int(elapsed_ms) % self.total_duration_ms
        acc = 0
        for idx, duration in enumerate(self.durations_ms):
            acc += max(1, int(duration))
            if target < acc:
                return idx
        return len(self.frames) - 1


@lru_cache(maxsize=1)
def _resolve_font_path() -> Optional[str]:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/trebucbd.ttf"),
        Path("C:/Windows/Fonts/verdanab.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


@lru_cache(maxsize=32)
def load_font(size: int) -> ImageFont.ImageFont:
    path = _resolve_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    return tuple(int(v) for v in draw.textsize(text, font=font))


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []
    words = stripped.split()
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        probe = f"{current} {word}"
        probe_width, _ = measure_text(draw, probe, font)
        if probe_width <= max_width:
            current = probe
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


class BabaSpriteLibrary:
    """Loads BabaGUI GIFs and resolves animated tile frames."""

    def __init__(self, asset_root: Optional[Path] = None):
        if asset_root is None:
            asset_root = Path(__file__).resolve().parents[2] / "assets" / "babagui_sprites"
        self.asset_root = Path(asset_root)
        self.icon_animations = self._load_animation_group(self.asset_root / "icon")
        self.text_animations = self._load_animation_group(self.asset_root / "text")
        self._scaled_cache: Dict[Tuple[str, str, int, int], Image.Image] = {}

    def _load_animation_group(self, directory: Path) -> Dict[str, SpriteAnimation]:
        animations: Dict[str, SpriteAnimation] = {}
        if not directory.exists():
            return animations
        for path in sorted(directory.glob("*.gif")):
            with Image.open(path) as img:
                frames: List[Image.Image] = []
                durations: List[int] = []
                for frame in ImageSequence.Iterator(img):
                    rgba = frame.convert("RGBA")
                    frames.append(rgba.copy())
                    durations.append(max(60, int(frame.info.get("duration") or img.info.get("duration") or 120)))
            if frames:
                animations[path.stem.lower()] = SpriteAnimation(
                    frames=tuple(frames),
                    durations_ms=tuple(durations),
                    total_duration_ms=sum(durations),
                )
        return animations

    def _resolve_scaled_frame(
        self,
        *,
        group: str,
        name: str,
        tile_size: int,
        elapsed_ms: int,
    ) -> Optional[Image.Image]:
        animations = self.text_animations if group == "text" else self.icon_animations
        animation = animations.get(normalize_word(name))
        if animation is None:
            return None
        frame_index = animation.resolve_frame_index(int(elapsed_ms))
        cache_key = (group, normalize_word(name), int(tile_size), int(frame_index))
        cached = self._scaled_cache.get(cache_key)
        if cached is not None:
            return cached
        frame = animation.frames[frame_index]
        scaled = frame.resize((int(tile_size), int(tile_size)), RESAMPLE_NEAREST)
        self._scaled_cache[cache_key] = scaled
        return scaled

    def render_base_tile(
        self,
        *,
        tile_size: int,
        elapsed_ms: int,
        fill_color: str = "#020202",
        outline_color: str = "#131313",
    ) -> Image.Image:
        tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)
        draw.rounded_rectangle(
            (2, 2, tile_size - 3, tile_size - 3),
            radius=max(4, tile_size // 7),
            fill=fill_color,
            outline=outline_color,
            width=1,
        )
        return tile

    def render_rule_tile(
        self,
        *,
        word: str,
        obj_type: str,
        tile_size: int,
        elapsed_ms: int,
    ) -> Image.Image:
        sprite = self._resolve_scaled_frame(
            group="text",
            name=word,
            tile_size=tile_size,
            elapsed_ms=elapsed_ms,
        )
        if sprite is not None:
            return sprite.copy()

        fill_color, outline_color, text_color = RULE_TILE_COLORS.get(
            normalize_word(obj_type),
            ("#1F1727", "#F8E38A", "#FFF8DB"),
        )
        tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)
        margin = max(2, tile_size // 12)
        draw.rounded_rectangle(
            (margin, margin, tile_size - margin - 1, tile_size - margin - 1),
            radius=max(4, tile_size // 8),
            fill=fill_color,
            outline=outline_color,
            width=max(1, tile_size // 16),
        )
        font = load_font(max(14, tile_size // 4))
        label = normalize_word(word).upper()
        chunks = [label]
        if len(label) >= 6:
            pivot = max(3, len(label) // 2)
            chunks = [label[:pivot], label[pivot:]]
        total_height = 0
        measured: List[Tuple[int, int]] = []
        for chunk in chunks:
            size = measure_text(draw, chunk, font)
            measured.append(size)
            total_height += size[1]
        total_height += max(0, (len(chunks) - 1) * max(1, tile_size // 18))
        cursor_y = (tile_size - total_height) // 2
        for idx, chunk in enumerate(chunks):
            text_w, text_h = measured[idx]
            cursor_x = (tile_size - text_w) // 2
            draw.text((cursor_x + 1, cursor_y + 1), chunk, font=font, fill="#100913")
            draw.text((cursor_x, cursor_y), chunk, font=font, fill=text_color)
            cursor_y += text_h + max(1, tile_size // 18)
        return tile

    def render_world_tile(
        self,
        *,
        word: str,
        tile_size: int,
        elapsed_ms: int,
        highlight_kind: Optional[str] = None,
        background_fill: str = "#020202",
        background_outline: str = "#131313",
    ) -> Image.Image:
        tile = self.render_base_tile(
            tile_size=tile_size,
            elapsed_ms=elapsed_ms,
            fill_color=background_fill,
            outline_color=background_outline,
        )
        sprite = self._resolve_scaled_frame(
            group="icon",
            name=word,
            tile_size=max(12, int(tile_size * 0.8)),
            elapsed_ms=elapsed_ms,
        )
        if sprite is not None:
            left = (tile_size - sprite.width) // 2
            top = (tile_size - sprite.height) // 2
            tile.alpha_composite(sprite, (left, top))
        else:
            self._draw_world_fallback(tile=tile, word=word)

        self._draw_highlight(tile=tile, highlight_kind=highlight_kind)
        return tile

    def _draw_world_fallback(self, *, tile: Image.Image, word: str) -> None:
        draw = ImageDraw.Draw(tile)
        tile_size = tile.size[0]
        main_color, accent_color = FALLBACK_WORLD_COLORS.get(
            normalize_word(word),
            ("#B7B3C5", "#F8F4FF"),
        )
        inset = max(5, tile_size // 10)
        normalized = normalize_word(word)
        if normalized == "door":
            draw.rounded_rectangle(
                (inset + 4, inset, tile_size - inset - 4, tile_size - inset),
                radius=max(4, tile_size // 8),
                fill=main_color,
                outline="#0B1020",
                width=max(1, tile_size // 16),
            )
            knob_r = max(2, tile_size // 18)
            knob_x = tile_size - inset - 10
            knob_y = tile_size // 2
            draw.ellipse(
                (knob_x - knob_r, knob_y - knob_r, knob_x + knob_r, knob_y + knob_r),
                fill="#0B1020",
            )
            return
        if normalized == "key":
            ring_r = max(6, tile_size // 5)
            ring_cx = tile_size // 3
            ring_cy = tile_size // 2
            draw.ellipse(
                (
                    ring_cx - ring_r,
                    ring_cy - ring_r,
                    ring_cx + ring_r,
                    ring_cy + ring_r,
                ),
                outline=main_color,
                width=max(2, tile_size // 12),
            )
            shaft_h = max(4, tile_size // 12)
            shaft_x0 = ring_cx + ring_r - 2
            shaft_x1 = tile_size - inset
            shaft_y = ring_cy - shaft_h // 2
            draw.rounded_rectangle(
                (shaft_x0, shaft_y, shaft_x1, shaft_y + shaft_h),
                radius=max(2, tile_size // 24),
                fill=main_color,
                outline="#0B1020",
                width=1,
            )
            tooth_w = max(3, tile_size // 10)
            draw.rectangle(
                (shaft_x1 - tooth_w * 2, shaft_y + shaft_h, shaft_x1 - tooth_w, shaft_y + shaft_h * 2),
                fill=main_color,
            )
            draw.rectangle(
                (shaft_x1 - tooth_w, shaft_y + shaft_h, shaft_x1, shaft_y + shaft_h * 2 + 1),
                fill=main_color,
            )
            return
        if normalized == "box":
            draw.rounded_rectangle(
                (inset, inset + 2, tile_size - inset, tile_size - inset + 2),
                radius=max(3, tile_size // 10),
                fill=main_color,
                outline="#0B1020",
                width=max(1, tile_size // 16),
            )
            draw.line((inset + 4, inset + 8, tile_size - inset - 4, tile_size - inset - 4), fill=accent_color, width=2)
            draw.line((inset + 4, tile_size - inset - 4, tile_size - inset - 4, inset + 8), fill=accent_color, width=2)
            return

        draw.rounded_rectangle(
            (inset, inset, tile_size - inset, tile_size - inset),
            radius=max(4, tile_size // 8),
            fill=main_color,
            outline="#0B1020",
            width=max(1, tile_size // 16),
        )
        font = load_font(max(14, tile_size // 4))
        label = normalized.upper()[:4]
        text_w, text_h = measure_text(draw, label, font)
        draw.text(
            ((tile_size - text_w) // 2, (tile_size - text_h) // 2),
            label,
            font=font,
            fill=accent_color,
        )

    def _draw_highlight(self, *, tile: Image.Image, highlight_kind: Optional[str]) -> None:
        if highlight_kind is None:
            return
        draw = ImageDraw.Draw(tile)
        tile_size = tile.size[0]
        if highlight_kind == "you":
            outline = "#F7D354"
        elif highlight_kind == "win":
            outline = "#FF8CB9"
        else:
            outline = "#FF746B"
        outline_rgb = tuple(int(outline[i : i + 2], 16) for i in (1, 3, 5))
        for idx, alpha in enumerate((110, 180, 255)):
            inset = max(1, idx * 2 + 1)
            draw.rounded_rectangle(
                (inset, inset, tile_size - inset - 1, tile_size - inset - 1),
                radius=max(4, tile_size // 8),
                outline=outline_rgb + (alpha,),
                width=1 if idx < 2 else max(1, tile_size // 18),
            )


class BabaManualPlayerRenderer:
    """Composes the board, HUD, active rules, and action feed into one image."""

    def __init__(self, asset_root: Optional[Path] = None, max_log_entries: int = 6):
        self.sprites = BabaSpriteLibrary(asset_root=asset_root)
        self.max_log_entries = max(4, int(max_log_entries))

    def render(
        self,
        *,
        state: Dict[str, Any],
        env_label: str,
        scenario_type: Optional[str],
        step_index: int,
        action_name: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        episode_done: bool,
        info_message: Optional[str] = None,
        log_entries: Optional[Sequence[ActionLogEntry]] = None,
        animation_ms: int = 0,
        meta_lines: Optional[Sequence[str]] = None,
        visual_config: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        visual_state = canonicalize_state_for_visualization(
            state,
            visual_config=visual_config,
        )
        width, height = extract_grid_size(visual_state)
        if width <= 0 or height <= 0:
            raise ValueError("invalid grid size in state")

        tile_size = self._resolve_tile_size(width=width, height=height)
        board_img = self._render_board(
            state=visual_state,
            tile_size=tile_size,
            animation_ms=animation_ms,
        )
        panel_width = max(360, tile_size * 5)
        header_height = 138
        header_top = 16
        active_rules = extract_rule_sentences(visual_state)
        property_map = extract_active_property_map(visual_state)
        panel_sections = [
            self._draw_status_panel(
                width=panel_width,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                property_map=property_map,
                meta_lines=tuple(meta_lines or ()),
            ),
            self._draw_rule_panel(width=panel_width, active_rules=active_rules),
            self._draw_controls_panel(width=panel_width),
            self._draw_log_panel(
                width=panel_width,
                entries=list(log_entries or ())[-self.max_log_entries :],
            ),
        ]
        panel_stack_height = sum(section.height for section in panel_sections) + (12 * (len(panel_sections) - 1))
        canvas_width = board_img.width + panel_width + 68
        canvas_height = max(
            header_top + header_height + board_img.height + 36,
            header_top + header_height + panel_stack_height + 28,
        )

        canvas = Image.new("RGBA", (canvas_width, canvas_height), BACKGROUND_TOP)
        draw = ImageDraw.Draw(canvas)

        header_rect = (18, header_top, canvas_width - 18, header_top + header_height)
        self._draw_header(
            canvas=canvas,
            draw=draw,
            rect=header_rect,
            env_label=env_label,
            scenario_type=scenario_type,
            step_index=step_index,
            action_name=action_name,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info_message=info_message,
        )

        board_x = 18
        board_y = header_rect[3] + 10
        canvas.alpha_composite(board_img, (board_x, board_y))

        panel_x = board_x + board_img.width + 18
        cursor_y = board_y
        for section in panel_sections:
            canvas.alpha_composite(section, (panel_x, cursor_y))
            cursor_y += section.height + 12

        if episode_done:
            self._draw_episode_overlay(
                canvas=canvas,
                board_origin=(board_x, board_y),
                board_size=board_img.size,
                terminated=terminated,
                truncated=truncated,
                reward=reward,
            )

        return canvas.convert("RGB")

    def _resolve_tile_size(self, *, width: int, height: int) -> int:
        longest = max(int(width), int(height))
        if longest <= 7:
            return 92
        if longest <= 10:
            return 76
        if longest <= 14:
            return 64
        if longest <= 18:
            return 54
        return 46

    def render_state_board(
        self,
        *,
        state: Dict[str, Any],
        tile_size: Optional[int] = None,
        animation_ms: int = 0,
        visual_config: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        visual_state = canonicalize_state_for_visualization(
            state,
            visual_config=visual_config,
        )
        width, height = extract_grid_size(visual_state)
        resolved_tile_size = (
            self._resolve_tile_size(width=width, height=height)
            if tile_size is None
            else int(tile_size)
        )
        scene = build_state_board_scene(visual_state)
        return self.render_scene_board(
            scene=scene,
            tile_size=resolved_tile_size,
            animation_ms=animation_ms,
        )

    def render_custom_map_board(
        self,
        *,
        spec: Dict[str, Any],
        tile_size: Optional[int] = None,
        animation_ms: int = 0,
    ) -> Image.Image:
        width = int(spec.get("width", 0) or 0)
        height = int(spec.get("height", 0) or 0)
        resolved_tile_size = (
            self._resolve_tile_size(width=width, height=height)
            if tile_size is None
            else int(tile_size)
        )
        scene = build_custom_map_board_scene(spec)
        return self.render_scene_board(
            scene=scene,
            tile_size=resolved_tile_size,
            animation_ms=animation_ms,
        )

    def render_scene_board(
        self,
        *,
        scene: BoardScene,
        tile_size: int,
        animation_ms: int = 0,
    ) -> Image.Image:
        return self._render_scene_board(
            scene=scene,
            tile_size=int(tile_size),
            animation_ms=animation_ms,
        )

    def _render_board(self, *, state: Dict[str, Any], tile_size: int, animation_ms: int) -> Image.Image:
        return self.render_state_board(
            state=state,
            tile_size=int(tile_size),
            animation_ms=animation_ms,
        )

    def _render_scene_board(
        self,
        *,
        scene: BoardScene,
        tile_size: int,
        animation_ms: int,
    ) -> Image.Image:
        width = int(scene.width)
        height = int(scene.height)
        padding = 22
        board_width = width * tile_size + padding * 2
        board_height = height * tile_size + padding * 2
        board = Image.new("RGBA", (board_width, board_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(board)

        draw.rounded_rectangle(
            (0, 0, board_width - 1, board_height - 1),
            radius=28,
            fill=PANEL_FILL,
            outline=PANEL_OUTLINE,
            width=3,
        )
        inner_rect = (padding - 8, padding - 8, board_width - padding + 7, board_height - padding + 7)
        draw.rounded_rectangle(
            inner_rect,
            radius=18,
            fill="#010101",
            outline="#171717",
            width=1,
        )

        by_cell: Dict[Tuple[int, int], SceneCell] = {
            (int(cell.x), int(cell.y)): cell for cell in scene.cells
        }

        for y in range(height):
            for x in range(width):
                px = padding + x * tile_size
                py = padding + y * tile_size
                base_tile = self.sprites.render_base_tile(tile_size=tile_size, elapsed_ms=animation_ms)
                board.alpha_composite(base_tile, (px, py))

                cell = by_cell.get((x, y))
                if cell is None or not cell.objects:
                    continue

                cell_objects = list(cell.objects)
                visible_objects = cell_objects[-3:]
                if len(visible_objects) == 1:
                    obj = visible_objects[0]
                    tile = self._render_object_tile(
                        obj=obj,
                        tile_size=tile_size,
                        animation_ms=animation_ms,
                    )
                    board.alpha_composite(tile, (px, py))
                    self._draw_direction_marker(
                        board=board,
                        origin=(px, py),
                        tile_size=tile_size,
                        direction=str(obj.direction or ""),
                    )
                    continue

                stack_offsets = [(-6, 6), (0, 0), (6, -6)]
                stack_size = max(28, int(tile_size * 0.78))
                for index, obj in enumerate(visible_objects):
                    ox, oy = stack_offsets[-len(visible_objects) + index]
                    tile = self._render_object_tile(
                        obj=obj,
                        tile_size=stack_size,
                        animation_ms=animation_ms,
                    )
                    offset_x = px + (tile_size - stack_size) // 2 + ox
                    offset_y = py + (tile_size - stack_size) // 2 + oy
                    board.alpha_composite(tile, (offset_x, offset_y))
                    if index == len(visible_objects) - 1:
                        self._draw_direction_marker(
                            board=board,
                            origin=(offset_x, offset_y),
                            tile_size=stack_size,
                            direction=str(obj.direction or ""),
                        )

                if len(cell_objects) > len(visible_objects):
                    badge_size = max(18, tile_size // 3)
                    bx = px + tile_size - badge_size - 3
                    by = py + 3
                    draw.rounded_rectangle(
                        (bx, by, bx + badge_size, by + badge_size),
                        radius=max(6, badge_size // 3),
                        fill="#050505",
                        outline=GRID_GLOW,
                        width=1,
                    )
                    font = load_font(max(12, badge_size - 8))
                    label = str(len(cell_objects))
                    text_w, text_h = measure_text(draw, label, font)
                    draw.text(
                        (bx + (badge_size - text_w) // 2, by + (badge_size - text_h) // 2 - 1),
                        label,
                        font=font,
                        fill=TEXT_PRIMARY,
                    )

        return board

    def _render_object_tile(
        self,
        *,
        obj: SceneObject,
        tile_size: int,
        animation_ms: int,
    ) -> Image.Image:
        if obj.kind == "rule":
            return self.sprites.render_rule_tile(
                word=str(obj.sprite_key or obj.word),
                obj_type=obj.obj_type,
                tile_size=tile_size,
                elapsed_ms=animation_ms,
            )
        return self.sprites.render_world_tile(
            word=str(obj.sprite_key or obj.word),
            tile_size=tile_size,
            elapsed_ms=animation_ms,
            highlight_kind=obj.highlight_kind,
        )

    def _draw_direction_marker(
        self,
        *,
        board: Image.Image,
        origin: Tuple[int, int],
        tile_size: int,
        direction: str,
    ) -> None:
        normalized = str(direction or "").strip().lower()
        if not normalized:
            return
        cx = origin[0] + tile_size - max(10, tile_size // 5)
        cy = origin[1] + tile_size - max(10, tile_size // 5)
        radius = max(7, tile_size // 7)
        badge = Image.new("RGBA", (radius * 2 + 4, radius * 2 + 4), (0, 0, 0, 0))
        badge_draw = ImageDraw.Draw(badge)
        badge_draw.ellipse((2, 2, radius * 2 + 1, radius * 2 + 1), fill="#050505", outline=GRID_GLOW, width=1)
        center = (radius + 2, radius + 2)
        delta = max(3, radius - 3)
        if "up" in normalized:
            points = [(center[0], center[1] - delta), (center[0] - delta, center[1] + delta), (center[0] + delta, center[1] + delta)]
        elif "down" in normalized:
            points = [(center[0], center[1] + delta), (center[0] - delta, center[1] - delta), (center[0] + delta, center[1] - delta)]
        elif "left" in normalized:
            points = [(center[0] - delta, center[1]), (center[0] + delta, center[1] - delta), (center[0] + delta, center[1] + delta)]
        else:
            points = [(center[0] + delta, center[1]), (center[0] - delta, center[1] - delta), (center[0] - delta, center[1] + delta)]
        badge_draw.polygon(points, fill="#F7D354")
        board.alpha_composite(badge, (cx - radius - 2, cy - radius - 2))

    def _draw_header(
        self,
        *,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        rect: Tuple[int, int, int, int],
        env_label: str,
        scenario_type: Optional[str],
        step_index: int,
        action_name: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        info_message: Optional[str],
    ) -> None:
        x0, y0, x1, y1 = rect
        panel = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rounded_rectangle(
            (0, 0, panel.width - 1, panel.height - 1),
            radius=18,
            fill=CARD_FILL,
            outline=PANEL_OUTLINE,
            width=1,
        )

        title_font = load_font(24)
        body_font = load_font(15)
        meta_font = load_font(14)
        info_font = load_font(14)
        panel_draw.text((18, 12), "PLAY BABA ENV", font=title_font, fill=TEXT_PRIMARY)

        left_max_width = max(260, panel.width - 360)
        env_lines = wrap_text(panel_draw, env_label, body_font, left_max_width)[:2]
        text_y = 44
        for line in env_lines:
            panel_draw.text((18, text_y), line, font=body_font, fill=HEADER_ACCENT)
            text_y += 16

        subtitle = scenario_type if scenario_type else "default or randomized"
        subtitle_lines = wrap_text(panel_draw, subtitle, meta_font, left_max_width)[:1]
        for line in subtitle_lines:
            panel_draw.text((18, text_y + 2), line, font=meta_font, fill=TEXT_MUTED)
            text_y += 14

        status_key = resolve_status_key(terminated=terminated, truncated=truncated, reward=reward)
        status_label = {
            "running": "RUNNING",
            "win": "YOU WIN",
            "terminated": "ENDED",
            "truncated": "TIME LIMIT",
        }[status_key]
        self._draw_badge(
            panel_draw,
            text=status_label,
            rect=(panel.width - 166, 14, panel.width - 18, 48),
            status_key=status_key,
        )

        stats_line = f"STEP {int(step_index):04d}   ACTION {str(action_name).upper()}   REWARD {float(reward):+0.3f}"
        stats_w, _ = measure_text(panel_draw, stats_line, meta_font)
        panel_draw.text((panel.width - stats_w - 20, 64), stats_line, font=meta_font, fill=TEXT_PRIMARY)

        if info_message:
            wrapped = wrap_text(panel_draw, info_message, info_font, panel.width - 40)
            if wrapped:
                panel_draw.text((18, panel.height - 26), wrapped[0], font=info_font, fill="#CFC7B7")

        canvas.alpha_composite(panel, (x0, y0))

    def _draw_status_panel(
        self,
        *,
        width: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        property_map: Dict[str, set[str]],
        meta_lines: Sequence[str],
    ) -> Image.Image:
        panel = self._make_panel(width=width, height=188, title="LEVEL STATUS")
        draw = ImageDraw.Draw(panel)
        font = load_font(16)
        value_font = load_font(20)

        cursor_y = 42
        summary = {
            "avatar": ", ".join(sorted(noun.upper() for noun, props in property_map.items() if "you" in props)) or "NONE",
            "goal": ", ".join(sorted(noun.upper() for noun, props in property_map.items() if "win" in props)) or "NONE",
        }
        state_label = {
            "running": "RUNNING",
            "win": "WIN",
            "terminated": "ENDED",
            "truncated": "TIME",
        }[resolve_status_key(terminated=terminated, truncated=truncated, reward=reward)]
        for label, value in (
            ("reward", f"{float(reward):+0.3f}"),
            ("state", state_label),
            ("you", summary["avatar"]),
            ("win", summary["goal"]),
        ):
            draw.text((18, cursor_y), label.upper(), font=font, fill=TEXT_MUTED)
            draw.text((110, cursor_y - 2), value, font=value_font if label in {"reward", "state"} else font, fill=TEXT_PRIMARY)
            cursor_y += 28

        scenario_font = load_font(14)
        info_lines = [str(line) for line in meta_lines[:2]]
        info_y = panel.height - 14 - len(info_lines) * 15
        for line in info_lines:
            draw.text((18, info_y), line, font=scenario_font, fill="#C7BDD6")
            info_y += 15

        return panel

    def _draw_rule_panel(self, *, width: int, active_rules: Sequence[str]) -> Image.Image:
        row_count = max(4, len(active_rules))
        panel = self._make_panel(width=width, height=52 + row_count * 28, title="ACTIVE RULES")
        draw = ImageDraw.Draw(panel)
        font = load_font(16)
        row_y = 42
        if not active_rules:
            draw.text((18, row_y), "NO PARSED RULES", font=font, fill=TEXT_MUTED)
            return panel
        for index, rule in enumerate(active_rules[:8]):
            fill = "#0B0B0B" if index % 2 == 0 else "#080808"
            draw.rounded_rectangle((14, row_y - 4, panel.width - 14, row_y + 18), radius=10, fill=fill)
            draw.text((24, row_y), rule, font=font, fill=TEXT_PRIMARY)
            row_y += 28
        return panel

    def _draw_controls_panel(self, *, width: int) -> Image.Image:
        panel = self._make_panel(width=width, height=222, title="CONTROLS")
        draw = ImageDraw.Draw(panel)
        key_font = load_font(15)
        desc_font = load_font(16)
        controls = [
            ("ARROWS", "move"),
            ("SPACE", "idle"),
            ("Z", "undo"),
            ("R", "reset / next scenario"),
            ("S", "save current frame"),
            ("Q / ESC", "quit"),
        ]
        row_y = 44
        for key_label, desc in controls:
            draw.rounded_rectangle((18, row_y - 4, 112, row_y + 18), radius=9, fill="#070707", outline="#2B2B2B", width=1)
            key_w, _ = measure_text(draw, key_label, key_font)
            draw.text((65 - key_w // 2, row_y), key_label, font=key_font, fill=TEXT_PRIMARY)
            draw.text((128, row_y), desc, font=desc_font, fill=TEXT_MUTED)
            row_y += 28
        return panel

    def _draw_log_panel(self, *, width: int, entries: Sequence[ActionLogEntry]) -> Image.Image:
        visible_entries = list(entries)[-self.max_log_entries :]
        row_count = self.max_log_entries
        panel = self._make_panel(width=width, height=58 + row_count * 42, title="ACTION FEED")
        draw = ImageDraw.Draw(panel)
        label_font = load_font(14)
        value_font = load_font(16)
        row_y = 42
        if not visible_entries:
            draw.text((18, row_y), "NO ACTIONS YET", font=value_font, fill=TEXT_MUTED)
            return panel
        for entry in reversed(visible_entries):
            draw.rounded_rectangle((14, row_y - 6, panel.width - 14, row_y + 28), radius=12, fill="#0A0A0A")
            self._draw_badge(
                draw,
                text=entry.label,
                rect=(20, row_y - 1, 108, row_y + 23),
                status_key=entry.status_key,
                compact=True,
            )
            draw.text((118, row_y - 1), f"STEP {int(entry.step_index):04d}", font=label_font, fill=TEXT_MUTED)
            draw.text((118, row_y + 15), entry.reward_text, font=value_font, fill=TEXT_PRIMARY)
            if entry.note:
                note = entry.note[:28]
                note_w, _ = measure_text(draw, note, label_font)
                draw.text((panel.width - note_w - 18, row_y + 8), note, font=label_font, fill="#B1AA9C")
            row_y += 42
        return panel

    def _draw_episode_overlay(
        self,
        *,
        canvas: Image.Image,
        board_origin: Tuple[int, int],
        board_size: Tuple[int, int],
        terminated: bool,
        truncated: bool,
        reward: float,
    ) -> None:
        board_x, board_y = board_origin
        board_w, board_h = board_size
        overlay = Image.new("RGBA", (board_w, board_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle((0, 0, board_w - 1, board_h - 1), radius=28, fill=(8, 8, 12, 130))
        banner_w = min(board_w - 48, 420)
        banner_h = 116
        banner_x = (board_w - banner_w) // 2
        banner_y = (board_h - banner_h) // 2
        status_key = resolve_status_key(terminated=terminated, truncated=truncated, reward=reward)
        fill, outline = STATUS_COLORS[status_key]
        draw.rounded_rectangle(
            (banner_x, banner_y, banner_x + banner_w, banner_y + banner_h),
            radius=24,
            fill=fill,
            outline=outline,
            width=3,
        )
        title_font = load_font(28)
        body_font = load_font(18)
        if status_key == "win":
            title = "YOU WIN"
            body = "PRESS R TO PLAY AGAIN"
        elif status_key == "truncated":
            title = "TIME LIMIT"
            body = "PRESS R TO RESET"
        else:
            title = "LEVEL ENDED"
            body = "PRESS R TO RESET"
        title_w, _ = measure_text(draw, title, title_font)
        body_w, _ = measure_text(draw, body, body_font)
        draw.text((banner_x + (banner_w - title_w) // 2, banner_y + 26), title, font=title_font, fill=TEXT_PRIMARY)
        draw.text((banner_x + (banner_w - body_w) // 2, banner_y + 68), body, font=body_font, fill="#FFF2B0")
        canvas.alpha_composite(overlay, (board_x, board_y))

    def _make_panel(self, *, width: int, height: int, title: str) -> Image.Image:
        panel = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (0, 0, width - 1, height - 1),
            radius=16,
            fill=CARD_FILL,
            outline=CARD_OUTLINE,
            width=1,
        )
        title_font = load_font(18)
        draw.text((18, 14), title, font=title_font, fill=TEXT_PRIMARY)
        return panel

    def _draw_badge(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        text: str,
        rect: Tuple[int, int, int, int],
        status_key: str,
        compact: bool = False,
    ) -> None:
        fill, outline = STATUS_COLORS.get(status_key, STATUS_COLORS["running"])
        x0, y0, x1, y1 = rect
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=11 if compact else 14,
            fill=fill,
            outline=outline,
            width=2 if not compact else 1,
        )
        font = load_font(14 if compact else 16)
        text_w, text_h = measure_text(draw, text, font)
        draw.text(
            (x0 + (x1 - x0 - text_w) // 2, y0 + (y1 - y0 - text_h) // 2 - 1),
            text,
            font=font,
            fill=TEXT_PRIMARY,
        )
