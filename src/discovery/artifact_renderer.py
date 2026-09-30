from __future__ import annotations

import base64
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

from src.web.map_display_names import resolve_map_display_name


class TransitionArtifactRenderer:
    _PLACEHOLDER_PNG_BYTES = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0uoAAAAASUVORK5CYII="
    )

    def __init__(self, *, visual_config: Optional[Dict[str, Any]] = None) -> None:
        self._warned_missing_pillow = False
        self._board_renderer = None
        self._visual_config = dict(visual_config or {})
        self._font_cache: Dict[tuple[int, bool], Any] = {}

    def render_state_snapshot(
        self,
        *,
        state: Dict[str, Any],
        title: str,
        outpath: Path,
        action_name: Optional[str] = None,
        show_context_text: bool = True,
        include_world_in_context: bool = True,
        visual_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        image = self.render_state_snapshot_image(
            state=state,
            title=title,
            action_name=action_name,
            show_context_text=show_context_text,
            include_world_in_context=include_world_in_context,
            visual_config=visual_config,
        )
        if image is not None:
            try:
                image.save(outpath, format="PNG")
            finally:
                image.close()
            return True
        return self._write_placeholder_png(outpath)

    def _write_placeholder_png(self, outpath: Path) -> bool:
        try:
            outpath.write_bytes(self._PLACEHOLDER_PNG_BYTES)
        except OSError:
            return False
        return True

    def render_state_snapshot_image(
        self,
        *,
        state: Dict[str, Any],
        title: str,
        action_name: Optional[str] = None,
        show_context_text: bool = True,
        include_world_in_context: bool = True,
        visual_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            if not self._warned_missing_pillow:
                print("  [WARN] Pillow is not installed; using placeholder transition snapshot.")
                self._warned_missing_pillow = True
            return None

        width, height = self.extract_grid_size(state)
        if width <= 0 or height <= 0:
            print("  [WARN] Invalid grid size for transition visualization; skipped PNG save.")
            return None

        renderer = self._get_board_renderer()
        resolved_visual_config = (
            dict(visual_config)
            if isinstance(visual_config, dict)
            else dict(self._visual_config)
        )
        board = renderer.render_state_board(
            state=state,
            animation_ms=0,
            visual_config=resolved_visual_config,
        )
        snapshot_meta = self.describe_state_snapshot(
            state=state,
            action_name=action_name,
            include_world_in_context=include_world_in_context,
        )
        title_text = str(title or "").strip() or "State Snapshot"
        snapshot_font_sizes = self._resolve_snapshot_font_sizes(board_width=board.width)
        context_text = self._format_snapshot_context_text(snapshot_meta) if bool(show_context_text) else ""
        chip_label_font = self._load_font(size=snapshot_font_sizes["chip_label"], bold=True)
        chip_value_font = self._load_font(size=snapshot_font_sizes["chip_value"], bold=True)
        context_font = self._load_font(size=snapshot_font_sizes["context"], bold=False)
        context_label_font = self._load_font(size=snapshot_font_sizes["context"] + 2, bold=True)
        title_font = self._load_font(size=snapshot_font_sizes["title"], bold=True)
        context_runs = self._build_context_text_runs(
            snapshot_meta=snapshot_meta,
            label_font=context_label_font,
            value_font=context_font,
        ) if context_text else []
        outer_pad = 18
        header_pad_x = 24
        header_pad_y = 20
        chip_gap_x = 12
        chip_gap_y = 10
        title_chip_gap = 26
        chip_context_gap = 14
        layout_target_width = max(int(board.width) + 40, 560)
        probe = Image.new("RGB", (1, 1), "#111111")
        probe_draw = ImageDraw.Draw(probe)
        detail_rows = self._layout_meta_chip_rows(
            draw=probe_draw,
            items=snapshot_meta.get("detailItems", []),
            label_font=chip_label_font,
            value_font=chip_value_font,
            max_width=layout_target_width,
            horizontal_gap=chip_gap_x,
        )
        title_width, title_height = self._measure_text(
            draw=probe_draw,
            text=title_text,
            font=title_font,
        )
        context_width, context_height = self._measure_text_runs(
            draw=probe_draw,
            runs=context_runs,
        ) if context_runs else (0, 0)
        detail_width = max((row["width"] for row in detail_rows), default=0)
        header_content_width = max(int(board.width), title_width, context_width, detail_width)
        canvas_width = max(
            int(board.width) + outer_pad * 2,
            header_content_width + header_pad_x * 2 + outer_pad * 2,
        )
        header_inner_width = canvas_width - (outer_pad * 2 + header_pad_x * 2)
        header_height = header_pad_y * 2 + title_height
        if detail_rows:
            header_height += self._chip_rows_height(detail_rows, vertical_gap=chip_gap_y)
            header_height += title_chip_gap
        if context_text:
            header_height += context_height
            if detail_rows:
                header_height += chip_context_gap
            else:
                header_height += title_chip_gap
        header_height = max(140, header_height)
        canvas_height = board.height + header_height + outer_pad * 2 + 12
        image = Image.new("RGB", (canvas_width, canvas_height), "#0A0A0A")
        draw = ImageDraw.Draw(image)
        header_bounds = (
            outer_pad,
            outer_pad,
            canvas_width - outer_pad,
            outer_pad + header_height,
        )
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle(
                header_bounds,
                radius=20,
                fill="#121820",
                outline="#283341",
                width=2,
            )
        else:
            draw.rectangle(header_bounds, fill="#121820", outline="#283341")
        header_left = outer_pad + header_pad_x
        cursor_y = outer_pad + header_pad_y
        title_x = header_left + max(0, (header_inner_width - title_width) // 2)
        draw.text((title_x, cursor_y - 2), title_text, fill="#F6F2EA", font=title_font)
        cursor_y += title_height
        if detail_rows:
            cursor_y += title_chip_gap
            cursor_y = self._draw_meta_chip_rows(
                draw=draw,
                rows=detail_rows,
                left=header_left,
                width=header_inner_width,
                start_y=cursor_y,
                horizontal_gap=chip_gap_x,
                vertical_gap=chip_gap_y,
                label_font=chip_label_font,
                value_font=chip_value_font,
            )
        if context_text:
            cursor_y += chip_context_gap if detail_rows else title_chip_gap
            context_x = header_left + max(0, (header_inner_width - context_width) // 2)
            self._draw_text_runs(
                draw=draw,
                runs=context_runs,
                left=context_x,
                top=cursor_y - 1,
            )
        divider_y = outer_pad + header_height + 3
        draw.line(
            (outer_pad + 6, divider_y, canvas_width - outer_pad - 6, divider_y),
            fill="#212A34",
            width=2,
        )
        board_x = (canvas_width - board.width) // 2
        board_y = outer_pad + header_height + 8
        image.paste(board.convert("RGB"), (board_x, board_y))
        board.close()
        return image

    def compose_image_card(self, *, accent_color: str, image, label: Optional[str] = None):
        from PIL import Image, ImageDraw

        pad = 14
        header_height = 0
        label_font = None
        if isinstance(label, str) and label.strip():
            label_font = self._load_font(size=18, bold=True)
            probe = Image.new("RGB", (1, 1), "#111111")
            probe_draw = ImageDraw.Draw(probe)
            _label_width, label_height = self._measure_text(
                draw=probe_draw,
                text=str(label or ""),
                font=label_font,
            )
            header_height = max(42, label_height + 18)
        width = int(image.width) + pad * 2
        height = header_height + int(image.height) + pad * 2
        card = Image.new("RGB", (width, height), "#0A0A0A")
        draw = ImageDraw.Draw(card)
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle(
                (0, 0, width - 1, height - 1),
                radius=18,
                fill="#0A0A0A",
                outline=accent_color,
                width=2,
            )
        else:
            draw.rectangle((0, 0, width - 1, height - 1), fill="#0A0A0A", outline=accent_color)
        if label_font is not None:
            draw.text(
                (pad, pad - 1),
                str(label or ""),
                fill=accent_color,
                font=label_font,
            )
            draw.line(
                (pad, header_height + 2, width - pad, header_height + 2),
                fill=accent_color,
                width=2,
            )
        image_x = (width - int(image.width)) // 2
        image_y = pad + header_height
        card.paste(image.convert("RGB"), (image_x, image_y))
        return card

    def compose_comparison_image(
        self,
        *,
        previous_state: Dict[str, Any],
        actual_next_state: Dict[str, Any],
        predicted_next_state: Optional[Dict[str, Any]],
        action_name: str,
        experiment_name: str,
        version_tag: str,
        prediction_error: Optional[str],
        step_index: int,
        difference_summary: Optional[str] = None,
        status_lines: Optional[list[str]] = None,
        include_world_in_context: bool = True,
        visual_config: Optional[Dict[str, Any]] = None,
    ):
        from PIL import Image, ImageDraw

        comparison_meta = self.describe_state_snapshot(
            state=previous_state,
            action_name=action_name,
            include_world_in_context=include_world_in_context,
        )
        comparison_context_text = self._format_snapshot_context_text(comparison_meta)
        actual_before = self.render_state_snapshot_image(
            state=previous_state,
            title="Previous State",
            action_name=action_name,
            show_context_text=False,
            include_world_in_context=include_world_in_context,
            visual_config=visual_config,
        )
        actual_after = self.render_state_snapshot_image(
            state=actual_next_state,
            title="Expected Next State",
            action_name=action_name,
            show_context_text=False,
            include_world_in_context=include_world_in_context,
            visual_config=visual_config,
        )
        predicted_after = (
            self.render_state_snapshot_image(
                state=predicted_next_state,
                title="Predicted Next State",
                action_name=action_name,
                show_context_text=False,
                include_world_in_context=include_world_in_context,
                visual_config=visual_config,
            )
            if isinstance(predicted_next_state, dict)
            else None
        )
        if actual_before is None or actual_after is None:
            raise RuntimeError("Unable to render comparison image panels.")

        try:
            if predicted_after is None:
                predicted_after = Image.new("RGB", actual_after.size, "#111111")
                draw = ImageDraw.Draw(predicted_after)
                draw.text(
                    (18, 18),
                    "prediction unavailable",
                    fill="#F3EFE6",
                    font=self._load_font(size=20, bold=True),
                )

            before_card = self.compose_image_card(
                accent_color="#87B8F4",
                image=actual_before,
            )
            actual_card = self.compose_image_card(
                accent_color="#F2B36D",
                image=actual_after,
            )
            predicted_card = self.compose_image_card(
                accent_color="#7ED7C2",
                image=predicted_after,
            )

            pad = 22
            row_gap = 18
            column_gap = 16
            chip_gap_x = 12
            chip_gap_y = 10
            title_chip_gap = 18
            chip_context_gap = 10
            context_summary_gap = 14
            title_font = self._load_font(size=34, bold=True)
            chip_label_font = self._load_font(size=21, bold=True)
            chip_value_font = self._load_font(size=24, bold=True)
            context_font = self._load_font(size=18, bold=False)
            context_label_font = self._load_font(size=20, bold=True)
            summary_label_font = self._load_font(size=22, bold=True)
            summary_value_font = self._load_font(size=20, bold=False)
            context_runs = self._build_context_text_runs(
                snapshot_meta=comparison_meta,
                label_font=context_label_font,
                value_font=context_font,
            ) if comparison_context_text else []

            bottom_row_width = int(actual_card.width) + int(predicted_card.width) + column_gap
            action_label = self._normalize_action_name(action_name)
            header_chip_items = []
            if action_label:
                header_chip_items.append(
                    {
                        "label": "ACTION",
                        "value": action_label,
                        "tone": "action",
                    }
                )
            header_chip_items.append(
                {
                    "label": "STEP",
                    "value": str(int(step_index)),
                    "tone": "neutral",
                }
            )
            probe = Image.new("RGB", (1, 1), "#111111")
            probe_draw = ImageDraw.Draw(probe)
            header_chip_rows = self._layout_meta_chip_rows(
                draw=probe_draw,
                items=header_chip_items,
                label_font=chip_label_font,
                value_font=chip_value_font,
                max_width=max(bottom_row_width, int(before_card.width), 760),
                horizontal_gap=chip_gap_x,
            )
            title_text = "Transition Comparison"
            header_text_width, title_height = self._measure_text(
                draw=probe_draw,
                text=title_text,
                font=title_font,
            )
            context_text_width, context_text_height = self._measure_text_runs(
                draw=probe_draw,
                runs=context_runs,
            ) if context_runs else (0, 0)
            summary_runs = self._build_prefixed_text_runs(
                text=str(difference_summary or "").strip(),
                label_font=summary_label_font,
                value_font=summary_value_font,
                max_width=max(int(before_card.width), bottom_row_width, 760) - 40,
                prefixes=("Diff:",),
            )
            summary_text_width, summary_text_height = self._measure_text_runs(
                draw=probe_draw,
                runs=summary_runs,
            ) if summary_runs else (0, 0)

            header_chip_width = max((row["width"] for row in header_chip_rows), default=0)
            content_width = max(
                int(before_card.width),
                bottom_row_width,
                header_text_width + 40,
                header_chip_width + 40,
                context_text_width + 40,
                summary_text_width + 40,
            )
            header_height = 24 + title_height
            if header_chip_rows:
                header_height += title_chip_gap
                header_height += self._chip_rows_height(header_chip_rows, vertical_gap=chip_gap_y)
            if comparison_context_text:
                header_height += context_text_height
                if header_chip_rows:
                    header_height += chip_context_gap
                else:
                    header_height += title_chip_gap
            if summary_runs:
                header_height += context_summary_gap
                header_height += summary_text_height
            header_height += 24
            content_height = int(before_card.height) + row_gap + max(int(actual_card.height), int(predicted_card.height))
            image = Image.new(
                "RGB",
                (content_width + pad * 2, header_height + content_height + pad * 2),
                "#050505",
            )
            draw = ImageDraw.Draw(image)
            header_bounds = (
                pad,
                pad,
                content_width + pad,
                pad + header_height,
            )
            if hasattr(draw, "rounded_rectangle"):
                draw.rounded_rectangle(
                    header_bounds,
                    radius=24,
                    fill="#121820",
                    outline="#2C3644",
                    width=2,
                )
            else:
                draw.rectangle(header_bounds, fill="#121820", outline="#2C3644")
            cursor_y = pad + 18
            title_x = pad + max(0, (content_width - header_text_width) // 2)
            draw.text((title_x, cursor_y - 2), title_text, fill="#F6F2EA", font=title_font)
            cursor_y += title_height
            if header_chip_rows:
                cursor_y += title_chip_gap
                cursor_y = self._draw_meta_chip_rows(
                    draw=draw,
                    rows=header_chip_rows,
                    left=pad + 20,
                    width=content_width - 40,
                    start_y=cursor_y,
                    horizontal_gap=chip_gap_x,
                    vertical_gap=chip_gap_y,
                    label_font=chip_label_font,
                    value_font=chip_value_font,
                )
            if summary_runs:
                if comparison_context_text:
                    cursor_y += chip_context_gap
                    context_x = pad + max(0, (content_width - context_text_width) // 2)
                    self._draw_text_runs(
                        draw=draw,
                        runs=context_runs,
                        left=context_x,
                        top=cursor_y - 1,
                    )
                    cursor_y += context_text_height
                cursor_y += context_summary_gap
                summary_x = pad + max(0, (content_width - summary_text_width) // 2)
                self._draw_text_runs(
                    draw=draw,
                    runs=summary_runs,
                    left=summary_x,
                    top=cursor_y - 1,
                )
            elif comparison_context_text:
                cursor_y += chip_context_gap if header_chip_rows else title_chip_gap
                context_x = pad + max(0, (content_width - context_text_width) // 2)
                self._draw_text_runs(
                    draw=draw,
                    runs=context_runs,
                    left=context_x,
                    top=cursor_y - 1,
                )

            before_x = pad + (content_width - int(before_card.width)) // 2
            before_y = pad + header_height + 10
            image.paste(before_card.convert("RGB"), (before_x, before_y))

            bottom_y = before_y + int(before_card.height) + row_gap
            bottom_row_x = pad + (content_width - bottom_row_width) // 2
            image.paste(actual_card.convert("RGB"), (bottom_row_x, bottom_y))
            image.paste(
                predicted_card.convert("RGB"),
                (bottom_row_x + int(actual_card.width) + column_gap, bottom_y),
            )
            return image
        finally:
            actual_before.close()
            actual_after.close()
            if predicted_after is not None:
                predicted_after.close()
            if "before_card" in locals():
                before_card.close()
            if "actual_card" in locals():
                actual_card.close()
            if "predicted_card" in locals():
                predicted_card.close()

    def describe_state_snapshot(
        self,
        *,
        state: Dict[str, Any],
        action_name: Optional[str] = None,
        include_world_in_context: bool = True,
    ) -> Dict[str, Any]:
        world_index = self._extract_state_world_index(state)
        world_label = self._extract_state_world_label(state)
        map_name = self._extract_state_map_name(state)
        terminated = self._extract_state_terminated(state)
        action_label = self._normalize_action_name(action_name)
        world_display = str(int(world_index)) if world_index is not None else world_label

        context_items = []
        if include_world_in_context and world_display:
            context_items.append({"label": "WORLD", "value": world_display, "tone": "world"})
        if map_name:
            context_items.append({"label": "MAP", "value": map_name, "tone": "map"})

        detail_items = []
        if action_label:
            detail_items.append({"label": "ACTION", "value": action_label, "tone": "action"})
        if terminated is not None:
            detail_items.append(
                {
                    "label": "DONE",
                    "value": str(bool(terminated)),
                    "tone": "done_true" if bool(terminated) else "done_false",
                }
            )

        return {
            "worldIndex": world_index,
            "worldLabel": world_label,
            "worldDisplay": world_display,
            "mapName": map_name,
            "action": action_label,
            "terminated": terminated,
            "done": terminated,
            "contextItems": context_items,
            "detailItems": detail_items,
            "contextLine": " | ".join(
                f"{item['label']} {item['value']}" for item in context_items
            ) if context_items else "",
            "detailLine": " | ".join(
                f"{item['label']} {item['value']}" for item in detail_items
            ) if detail_items else "",
        }

    def _get_board_renderer(self) -> Any:
        if self._board_renderer is None:
            from src.ui.baba_manual_player_ui import BabaManualPlayerRenderer

            self._board_renderer = BabaManualPlayerRenderer()
        return self._board_renderer

    def _measure_text(self, *, draw: Any, text: str, font: Any) -> tuple[int, int]:
        left, top, right, bottom = self._measure_text_bounds(draw=draw, text=text, font=font)
        return int(right - left), int(bottom - top)

    def _measure_text_bounds(self, *, draw: Any, text: str, font: Any) -> tuple[int, int, int, int]:
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return int(left), int(top), int(right), int(bottom)
        width, height = draw.textsize(text, font=font)
        return 0, 0, int(width), int(height)

    def _ellipsize_text(self, *, draw: Any, text: str, font: Any, max_width: int) -> str:
        raw_text = str(text or "").strip()
        if not raw_text:
            return ""
        if max_width <= 0:
            return raw_text
        text_width, _text_height = self._measure_text(draw=draw, text=raw_text, font=font)
        if text_width <= max_width:
            return raw_text
        ellipsis = "..."
        low = 0
        high = len(raw_text)
        best = ellipsis
        while low <= high:
            mid = (low + high) // 2
            candidate = raw_text[:mid].rstrip()
            rendered = f"{candidate}{ellipsis}"
            rendered_width, _ = self._measure_text(draw=draw, text=rendered, font=font)
            if rendered_width <= max_width:
                best = rendered
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _load_font(self, *, size: int, bold: bool) -> Any:
        from PIL import ImageFont

        resolved_size = max(11, int(size))
        cache_key = (resolved_size, bool(bold))
        cached = self._font_cache.get(cache_key)
        if cached is not None:
            return cached

        font_candidates = (
            [
                "DejaVuSans-Bold.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
            ]
            if bold
            else [
                "DejaVuSans.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/arial.ttf",
            ]
        )
        for candidate in font_candidates:
            try:
                font = ImageFont.truetype(candidate, resolved_size)
                self._font_cache[cache_key] = font
                return font
            except OSError:
                continue

        font = ImageFont.load_default()
        self._font_cache[cache_key] = font
        return font

    def _resolve_snapshot_font_sizes(self, *, board_width: int) -> Dict[str, int]:
        base_size = max(20, min(30, int(board_width // 10)))
        return {
            "chip_label": max(16, base_size - 4),
            "chip_value": base_size,
            "context": max(16, base_size - 6),
            "title": base_size + 12,
        }

    def _extract_state_world_index(self, state: Dict[str, Any]) -> Optional[int]:
        candidates = [
            state.get("world_index"),
            state.get("worldIndex"),
        ]
        step_payload = state.get("step")
        if isinstance(step_payload, dict):
            candidates.extend(
                [
                    step_payload.get("world_index"),
                    step_payload.get("worldIndex"),
                ]
            )
        for candidate in candidates:
            if isinstance(candidate, int) and int(candidate) > 0:
                return int(candidate)
        return None

    def _extract_state_terminated(self, state: Dict[str, Any]) -> Optional[bool]:
        step_payload = state.get("step")
        if isinstance(step_payload, dict) and "terminated" in step_payload:
            return bool(step_payload.get("terminated"))
        return None

    def _extract_state_world_label(self, state: Dict[str, Any]) -> Optional[str]:
        for candidate in (
            state.get("world_label"),
            state.get("worldLabel"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return str(candidate).strip()
        step_payload = state.get("step")
        if isinstance(step_payload, dict):
            for candidate in (
                step_payload.get("world_label"),
                step_payload.get("worldLabel"),
            ):
                if isinstance(candidate, str) and candidate.strip():
                    return str(candidate).strip()
        inferred = self._infer_world_label_from_state(state)
        return inferred or None

    def _normalize_action_name(self, action_name: Optional[str]) -> Optional[str]:
        if not isinstance(action_name, str):
            return None
        normalized = action_name.strip().upper()
        return normalized or None

    def _format_snapshot_context_text(self, snapshot_meta: Mapping[str, Any]) -> str:
        parts = []
        context_items = snapshot_meta.get("contextItems")
        if isinstance(context_items, list):
            for item in context_items:
                if not isinstance(item, Mapping):
                    continue
                label = str(item.get("label") or "").strip().title()
                value = str(item.get("value") or "").strip()
                if label and value:
                    parts.append(f"{label} {value}")
        return "  ·  ".join(parts)

    def _build_context_text_parts(self, snapshot_meta: Mapping[str, Any]) -> list[Dict[str, str]]:
        parts: list[Dict[str, str]] = []
        context_items = snapshot_meta.get("contextItems")
        if not isinstance(context_items, list):
            return parts
        for item in context_items:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label") or "").strip().upper()
            value = str(item.get("value") or "").strip()
            if not label or not value:
                continue
            if parts:
                parts.append({"kind": "separator", "text": "  ·  "})
            parts.append({"kind": "label", "text": label})
            parts.append({"kind": "value", "text": f" {value}"})
        return parts

    def _split_prefixed_text(self, text: str, *, prefixes: tuple[str, ...]) -> tuple[str, str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return "", ""
        for prefix in prefixes:
            if normalized_text.startswith(prefix):
                return prefix, normalized_text[len(prefix):].lstrip()
        return "", normalized_text

    def _build_context_text_runs(
        self,
        *,
        snapshot_meta: Mapping[str, Any],
        label_font: Any,
        value_font: Any,
    ) -> list[Dict[str, Any]]:
        return self._build_styled_text_runs(
            parts=self._build_context_text_parts(snapshot_meta),
            label_font=label_font,
            value_font=value_font,
            label_fill="#D3E3F7",
            value_fill="#8FA2B8",
            separator_fill="#5F6E81",
        )

    def _build_prefixed_text_runs(
        self,
        *,
        text: str,
        label_font: Any,
        value_font: Any,
        max_width: int,
        prefixes: tuple[str, ...],
    ) -> list[Dict[str, Any]]:
        from PIL import Image, ImageDraw

        prefix, remainder = self._split_prefixed_text(text, prefixes=prefixes)
        if not prefix and not remainder:
            return []
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1), "#111111"))
        if not prefix:
            resolved_text = self._ellipsize_text(
                draw=probe,
                text=remainder,
                font=value_font,
                max_width=max_width,
            )
            return [{"text": resolved_text, "font": value_font, "fill": "#E2D8BF"}]

        prefix_width, _prefix_height = self._measure_text(draw=probe, text=prefix, font=label_font)
        remaining_width = max(0, int(max_width) - prefix_width - 6)
        resolved_remainder = self._ellipsize_text(
            draw=probe,
            text=remainder,
            font=value_font,
            max_width=remaining_width,
        )
        parts = [{"kind": "label", "text": prefix}]
        if resolved_remainder:
            parts.append({"kind": "value", "text": f" {resolved_remainder}"})
        return self._build_styled_text_runs(
            parts=parts,
            label_font=label_font,
            value_font=value_font,
            label_fill="#F1D59A",
            value_fill="#E6DDCA",
            separator_fill="#E6DDCA",
        )

    def _build_styled_text_runs(
        self,
        *,
        parts: list[Dict[str, str]],
        label_font: Any,
        value_font: Any,
        label_fill: str,
        value_fill: str,
        separator_fill: str,
    ) -> list[Dict[str, Any]]:
        runs: list[Dict[str, Any]] = []
        for part in parts:
            kind = str(part.get("kind") or "").strip().lower()
            text = str(part.get("text") or "")
            if not text:
                continue
            if kind == "label":
                runs.append({"text": text, "font": label_font, "fill": label_fill})
            elif kind == "separator":
                runs.append({"text": text, "font": value_font, "fill": separator_fill})
            else:
                runs.append({"text": text, "font": value_font, "fill": value_fill})
        return runs

    def _measure_text_runs(self, *, draw: Any, runs: list[Dict[str, Any]]) -> tuple[int, int]:
        total_width = 0
        max_height = 0
        for run in runs:
            left, top, right, bottom = self._measure_text_bounds(
                draw=draw,
                text=str(run.get("text") or ""),
                font=run.get("font"),
            )
            total_width += int(right - left)
            max_height = max(max_height, int(bottom - top))
        return total_width, max_height

    def _draw_text_runs(
        self,
        *,
        draw: Any,
        runs: list[Dict[str, Any]],
        left: int,
        top: int,
    ) -> None:
        if not runs:
            return
        _line_width, line_height = self._measure_text_runs(draw=draw, runs=runs)
        cursor_x = int(left)
        for run in runs:
            text = str(run.get("text") or "")
            font = run.get("font")
            fill = str(run.get("fill") or "#F3EFE6")
            bounds = self._measure_text_bounds(draw=draw, text=text, font=font)
            run_height = int(bounds[3] - bounds[1])
            draw_x = cursor_x - int(bounds[0])
            draw_y = int(top) + max(0, (line_height - run_height) // 2) - int(bounds[1])
            draw.text((draw_x, draw_y), text, fill=fill, font=font)
            cursor_x += int(bounds[2] - bounds[0])

    def _chip_rows_height(self, rows: list[Dict[str, Any]], *, vertical_gap: int) -> int:
        if not rows:
            return 0
        return sum(int(row["height"]) for row in rows) + vertical_gap * max(0, len(rows) - 1)

    def _layout_meta_chip_rows(
        self,
        *,
        draw: Any,
        items: list[Dict[str, Any]],
        label_font: Any,
        value_font: Any,
        max_width: int,
        horizontal_gap: int,
    ) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        current_chips: list[Dict[str, Any]] = []
        current_width = 0
        current_height = 0
        resolved_max_width = max(220, int(max_width))
        for item in items:
            chip = dict(item)
            chip_width, chip_height = self._measure_meta_chip(
                draw=draw,
                label=str(chip.get("label") or ""),
                value=str(chip.get("value") or ""),
                label_font=label_font,
                value_font=value_font,
            )
            chip["width"] = chip_width
            chip["height"] = chip_height
            projected_width = chip_width if not current_chips else current_width + horizontal_gap + chip_width
            if current_chips and projected_width > resolved_max_width:
                rows.append(
                    {
                        "chips": current_chips,
                        "width": current_width,
                        "height": current_height,
                    }
                )
                current_chips = [chip]
                current_width = chip_width
                current_height = chip_height
                continue
            current_chips.append(chip)
            current_width = projected_width if len(current_chips) > 1 else chip_width
            current_height = max(current_height, chip_height)
        if current_chips:
            rows.append(
                {
                    "chips": current_chips,
                    "width": current_width,
                    "height": current_height,
                }
            )
        return rows

    def _draw_meta_chip_rows(
        self,
        *,
        draw: Any,
        rows: list[Dict[str, Any]],
        left: int,
        width: int,
        start_y: int,
        horizontal_gap: int,
        vertical_gap: int,
        label_font: Any,
        value_font: Any,
    ) -> int:
        cursor_y = int(start_y)
        for row in rows:
            row_width = int(row["width"])
            chip_x = int(left) + max(0, (int(width) - row_width) // 2)
            row_height = int(row["height"])
            for chip in row["chips"]:
                chip_height = int(chip["height"])
                chip_y = cursor_y + max(0, (row_height - chip_height) // 2)
                self._draw_meta_chip(
                    draw=draw,
                    left=chip_x,
                    top=chip_y,
                    label=str(chip.get("label") or ""),
                    value=str(chip.get("value") or ""),
                    tone=str(chip.get("tone") or "neutral"),
                    label_font=label_font,
                    value_font=value_font,
                )
                chip_x += int(chip["width"]) + int(horizontal_gap)
            cursor_y += row_height + int(vertical_gap)
        return cursor_y - (int(vertical_gap) if rows else 0)

    def _measure_meta_chip(
        self,
        *,
        draw: Any,
        label: str,
        value: str,
        label_font: Any,
        value_font: Any,
    ) -> tuple[int, int]:
        label_left, label_top, label_right, label_bottom = self._measure_text_bounds(
            draw=draw,
            text=label,
            font=label_font,
        )
        value_left, value_top, value_right, value_bottom = self._measure_text_bounds(
            draw=draw,
            text=value,
            font=value_font,
        )
        label_width = int(label_right - label_left)
        label_height = int(label_bottom - label_top)
        value_width = int(value_right - value_left)
        value_height = int(value_bottom - value_top)
        pad_x = 16
        pad_y = 11
        separator_gap = 10
        separator_width = 2
        content_width = label_width + separator_gap + separator_width + separator_gap + value_width
        chip_width = pad_x * 2 + content_width
        chip_height = pad_y * 2 + max(label_height, value_height)
        return chip_width, chip_height

    def _draw_meta_chip(
        self,
        *,
        draw: Any,
        left: int,
        top: int,
        label: str,
        value: str,
        tone: str,
        label_font: Any,
        value_font: Any,
    ) -> None:
        chip_width, chip_height = self._measure_meta_chip(
            draw=draw,
            label=label,
            value=value,
            label_font=label_font,
            value_font=value_font,
        )
        palette = self._resolve_chip_palette(tone)
        right = int(left) + chip_width
        bottom = int(top) + chip_height
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle(
                (left, top, right, bottom),
                radius=max(16, chip_height // 2),
                fill=palette["fill"],
                outline=palette["outline"],
                width=2,
            )
        else:
            draw.rectangle((left, top, right, bottom), fill=palette["fill"], outline=palette["outline"])
        pad_x = 16
        separator_gap = 10
        separator_width = 2
        label_left, label_top, label_right, label_bottom = self._measure_text_bounds(
            draw=draw,
            text=label,
            font=label_font,
        )
        value_left, value_top, value_right, value_bottom = self._measure_text_bounds(
            draw=draw,
            text=value,
            font=value_font,
        )
        label_width = int(label_right - label_left)
        label_height = int(label_bottom - label_top)
        value_width = int(value_right - value_left)
        value_height = int(value_bottom - value_top)
        content_width = label_width + separator_gap + separator_width + separator_gap + value_width
        content_height = max(label_height, value_height)
        content_left = int(left) + max(0, (chip_width - content_width) // 2)
        content_top = int(top) + max(0, (chip_height - content_height) // 2)
        label_x = content_left - label_left
        label_y = content_top + max(0, (content_height - label_height) // 2) - label_top
        draw.text((label_x, label_y), label, fill=palette["label"], font=label_font)
        separator_x = content_left + label_width + separator_gap
        separator_inner_height = max(12, chip_height - 18)
        separator_top = int(top) + max(0, (chip_height - separator_inner_height) // 2)
        separator_bottom = separator_top + separator_inner_height
        draw.line(
            (separator_x, separator_top, separator_x, separator_bottom),
            fill=palette["outline"],
            width=separator_width,
        )
        value_x = separator_x + separator_width + separator_gap - value_left
        value_y = content_top + max(0, (content_height - value_height) // 2) - value_top
        draw.text((value_x, value_y), value, fill=palette["value"], font=value_font)

    def _resolve_chip_palette(self, tone: str) -> Dict[str, str]:
        palettes = {
            "world": {
                "fill": "#101B27",
                "outline": "#72B9FF",
                "label": "#72B9FF",
                "value": "#F5FAFF",
            },
            "map": {
                "fill": "#10201F",
                "outline": "#7FD1C4",
                "label": "#7FD1C4",
                "value": "#F2FCFA",
            },
            "action": {
                "fill": "#251A10",
                "outline": "#F0B36C",
                "label": "#F0B36C",
                "value": "#FFF5E8",
            },
            "done_true": {
                "fill": "#132317",
                "outline": "#7FD67F",
                "label": "#7FD67F",
                "value": "#F1FFF1",
            },
            "done_false": {
                "fill": "#291515",
                "outline": "#E08989",
                "label": "#E08989",
                "value": "#FFF1F1",
            },
            "neutral": {
                "fill": "#181D24",
                "outline": "#9DA7B5",
                "label": "#B8C2D0",
                "value": "#F1F4F8",
            },
        }
        return palettes.get(str(tone or "neutral"), palettes["neutral"])

    def _infer_world_label_from_state(self, state: Dict[str, Any]) -> Optional[str]:
        candidates = [
            state.get("source"),
            state.get("artifact_stem"),
            state.get("artifactStem"),
            state.get("world_source"),
            state.get("worldSource"),
            state.get("requested_scenario_type"),
            state.get("scenario_type"),
            state.get("map_name"),
            state.get("mapName"),
        ]
        step_payload = state.get("step")
        if isinstance(step_payload, dict):
            candidates.extend(
                [
                    step_payload.get("source"),
                    step_payload.get("artifact_stem"),
                    step_payload.get("artifactStem"),
                    step_payload.get("world_source"),
                    step_payload.get("worldSource"),
                ]
            )
        for candidate in candidates:
            inferred = self._infer_world_label_from_text(candidate)
            if inferred:
                return inferred
        return None

    def _infer_world_label_from_text(self, raw_text: Any) -> Optional[str]:
        if not isinstance(raw_text, str):
            return None
        text = raw_text.strip()
        if not text:
            return None
        explicit_match = re.search(r"(?:world|env)[^0-9]{0,4}(\d+)", text, flags=re.IGNORECASE)
        if explicit_match:
            return str(int(explicit_match.group(1)))
        map_match = re.search(r"(?:map|scenario)[^0-9]{0,4}(\d+)", text, flags=re.IGNORECASE)
        if map_match:
            return str(int(map_match.group(1)))
        trailing_match = re.search(r"[_-](\d+)$", text)
        if trailing_match:
            return str(int(trailing_match.group(1)))
        return None

    def _extract_state_map_name(self, state: Dict[str, Any]) -> Optional[str]:
        resolved_map_name = resolve_map_display_name(
            state.get("map_name"),
            state.get("mapName"),
            state.get("scenario_type"),
            state.get("requested_scenario_type"),
        )
        if resolved_map_name:
            return resolved_map_name
        step_payload = state.get("step")
        if isinstance(step_payload, dict):
            resolved_map_name = resolve_map_display_name(
                step_payload.get("map_name"),
                step_payload.get("mapName"),
                step_payload.get("scenario_type"),
                step_payload.get("requested_scenario_type"),
            )
            if resolved_map_name:
                return resolved_map_name
        return None

    def extract_grid_size(self, state: Dict[str, Any]) -> tuple[int, int]:
        grid_size = state.get("grid_size")
        if (
            isinstance(grid_size, (list, tuple))
            and len(grid_size) == 2
            and isinstance(grid_size[0], int)
            and isinstance(grid_size[1], int)
            and grid_size[0] > 0
            and grid_size[1] > 0
        ):
            return int(grid_size[0]), int(grid_size[1])

        max_x = -1
        max_y = -1
        objects = state.get("objects")
        if isinstance(objects, list):
            for obj in objects:
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

        agent = state.get("agent")
        if isinstance(agent, dict):
            pos = agent.get("position")
            if (
                isinstance(pos, (list, tuple))
                and len(pos) == 2
                and isinstance(pos[0], int)
                and isinstance(pos[1], int)
            ):
                max_x = max(max_x, int(pos[0]))
                max_y = max(max_y, int(pos[1]))

        if max_x >= 0 and max_y >= 0:
            return max_x + 1, max_y + 1
        return 0, 0
