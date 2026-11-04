"""Sprite-driven Baba-style board renderer for discovery artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageSequence

from .board_scene import (
    BoardScene,
    SceneCell,
    SceneObject,
    build_custom_map_board_scene,
    build_state_board_scene,
    extract_grid_size,
    normalize_word,
)
from src.data.state_projection import canonicalize_state_for_artifact


RESAMPLE_NEAREST = Image.Resampling.NEAREST


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
TEXT_PRIMARY = "#F3EFE6"
GRID_GLOW = "#D7CFBF"


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


class BabaSpriteBoardRenderer:
    """Composes the board, HUD, active rules, and action feed into one image."""

    def __init__(self, asset_root: Optional[Path] = None):
        self.sprites = BabaSpriteLibrary(asset_root=asset_root)

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
        visual_state = canonicalize_state_for_artifact(
            state,
            state_projection_config=visual_config,
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
