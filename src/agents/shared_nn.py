from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from src.data.state_store import StateStore
from src.program_model.state_codec import parse_state_json
from src.visualization import canonicalize_visual_word


TYPE_NAMES = (
    "rule_noun",
    "rule_operator",
    "rule_property",
    "world_object",
)
TYPE_TO_INDEX = {name: index for index, name in enumerate(TYPE_NAMES)}

DEFAULT_WORD_VOCAB = (
    "baba",
    "and",
    "wall",
    "rock",
    "flag",
    "key",
    "door",
    "lava",
    "algae",
    "bog",
    "bolt",
    "brick",
    "bubble",
    "cog",
    "crab",
    "flower",
    "grass",
    "hedge",
    "ice",
    "jelly",
    "keke",
    "love",
    "pillar",
    "pipe",
    "reed",
    "skull",
    "star",
    "text",
    "tile",
    "water",
    "robot",
    "is",
    "you",
    "win",
    "stop",
    "push",
    "move",
    "open",
    "shift",
    "shut",
    "sink",
    "float",
    "hot",
    "melt",
    "defeat",
)

DIRECTION_NAMES = (
    "facing right",
    "facing down",
    "facing left",
    "facing up",
    "unknown",
)
DIRECTION_TO_INDEX = {name: index for index, name in enumerate(DIRECTION_NAMES)}

TOKEN_DIM = len(TYPE_NAMES) + len(DEFAULT_WORD_VOCAB) + len(DIRECTION_NAMES) + 4


class EntityTokenCodec:
    """Serialize Baba object sets into fixed-width entity tokens."""

    def __init__(
        self,
        word_vocab: Optional[Sequence[str]] = None,
        *,
        visual_config: Optional[Mapping[str, Any]] = None,
    ):
        normalized = [str(word).strip().lower() for word in (word_vocab or DEFAULT_WORD_VOCAB)]
        normalized = [word for word in normalized if word]
        if not normalized:
            raise ValueError("EntityTokenCodec requires at least one supported word.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("EntityTokenCodec word vocabulary must not contain duplicates.")
        self.word_vocab = tuple(normalized)
        self.word_to_index = {word: index for index, word in enumerate(self.word_vocab)}
        self.visual_config = dict(visual_config) if isinstance(visual_config, Mapping) else None
        self.token_dim = len(TYPE_NAMES) + len(self.word_vocab) + len(DIRECTION_NAMES) + 4
        self._type_index_cache: Dict[str, int] = {}
        self._word_index_cache: Dict[Tuple[str, str], int] = {}
        self._direction_index_cache: Dict[str, int] = {}

    def encode_state(self, state: Any) -> Tuple[np.ndarray, np.ndarray]:
        parsed = dict(state) if isinstance(state, dict) else parse_state_json(str(state))
        width, height = self._grid_size(parsed)
        raw_objects = parsed.get("objects", [])
        object_rows = raw_objects if isinstance(raw_objects, list) else []
        rows = np.zeros((max(1, len(object_rows)), self.token_dim), dtype=np.float32)
        row_count = 0
        for raw in object_rows:
            if not isinstance(raw, dict):
                continue
            position = raw.get("position")
            if (
                not isinstance(position, list)
                or len(position) != 2
                or not isinstance(position[0], int)
                or not isinstance(position[1], int)
            ):
                continue

            x = int(position[0])
            y = int(position[1])
            token = rows[row_count]

            obj_type = str(raw.get("type", "world_object")).strip().lower() or "world_object"
            type_index = self._type_index(obj_type)
            token[type_index] = 1.0

            serialized_word = str(raw.get("word", obj_type)).strip().lower() or obj_type
            word_index = self._word_index(obj_type=obj_type, serialized_word=serialized_word)
            token[len(TYPE_NAMES) + word_index] = 1.0

            direction = str(raw.get("direction", "unknown")).strip().lower() or "unknown"
            direction_index = self._direction_index(direction)
            direction_offset = len(TYPE_NAMES) + len(self.word_vocab)
            token[direction_offset + direction_index] = 1.0

            coord_offset = direction_offset + len(DIRECTION_NAMES)
            token[coord_offset + 0] = self._centered_coord(x, width)
            token[coord_offset + 1] = self._centered_coord(y, height)
            token[coord_offset + 2] = self._axis_size(width)
            token[coord_offset + 3] = self._axis_size(height)
            row_count += 1

        rows = rows[: max(1, row_count)]
        mask = np.ones((rows.shape[0],), dtype=np.bool_)
        return rows, mask

    def encode_state_id(
        self,
        state_store: StateStore,
        state_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        view = state_store.state_token_view(int(state_id))
        rows = np.zeros((max(1, view.object_count), self.token_dim), dtype=np.float32)
        row_count = 0

        for object_index in range(
            int(view.object_offset),
            int(view.object_offset) + int(view.object_count),
        ):
            object_flags = int(view.object_flags[object_index])
            if (
                object_flags & int(view.required_object_flags)
            ) != int(view.required_object_flags):
                continue

            type_id = int(view.object_type_ids[object_index])
            word_id = int(view.object_word_ids[object_index])
            if type_id < 0 or type_id >= len(view.type_texts):
                raise KeyError(f"State {int(state_id)} references unknown type id: {type_id}")
            if word_id < 0 or word_id >= len(view.word_texts):
                raise KeyError(f"State {int(state_id)} references unknown word id: {word_id}")

            token = rows[row_count]
            obj_type = str(view.type_texts[type_id]).strip().lower() or "world_object"
            type_index = self._type_index(obj_type)
            token[type_index] = 1.0

            serialized_word = str(view.word_texts[word_id]).strip().lower() or obj_type
            word_index = self._word_index(obj_type=obj_type, serialized_word=serialized_word)
            token[len(TYPE_NAMES) + word_index] = 1.0

            direction = "unknown"
            direction_id = int(view.object_direction_ids[object_index])
            if (
                object_flags & int(view.direction_object_flag)
                and direction_id != int(view.null_sentinel)
            ):
                if direction_id < 0 or direction_id >= len(view.direction_texts):
                    raise KeyError(
                        f"State {int(state_id)} references unknown direction id: {direction_id}"
                    )
                direction = str(view.direction_texts[direction_id]).strip().lower() or "unknown"
            direction_index = self._direction_index(direction)
            direction_offset = len(TYPE_NAMES) + len(self.word_vocab)
            token[direction_offset + direction_index] = 1.0

            x = int(view.object_x[object_index])
            y = int(view.object_y[object_index])
            coord_offset = direction_offset + len(DIRECTION_NAMES)
            token[coord_offset + 0] = self._centered_coord(x, int(view.width))
            token[coord_offset + 1] = self._centered_coord(y, int(view.height))
            token[coord_offset + 2] = self._axis_size(int(view.width))
            token[coord_offset + 3] = self._axis_size(int(view.height))
            row_count += 1

        rows = rows[: max(1, row_count)]
        mask = np.ones((rows.shape[0],), dtype=np.bool_)
        return rows, mask

    def _type_index(self, obj_type: str) -> int:
        cached = self._type_index_cache.get(obj_type)
        if cached is not None:
            return int(cached)
        resolved = TYPE_TO_INDEX.get(obj_type, TYPE_TO_INDEX["world_object"])
        self._type_index_cache[obj_type] = int(resolved)
        return int(resolved)

    def _word_index(self, *, obj_type: str, serialized_word: str) -> int:
        cache_key = (obj_type, serialized_word)
        cached = self._word_index_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        canonical_word = canonicalize_visual_word(
            obj_type,
            serialized_word,
            self.visual_config,
        )
        word_index = self.word_to_index.get(canonical_word)
        if word_index is None:
            raise ValueError(
                "Unsupported serialized state word "
                f"{serialized_word!r} (canonical={canonical_word!r}, type={obj_type!r})."
            )
        self._word_index_cache[cache_key] = int(word_index)
        return int(word_index)

    def _direction_index(self, direction: str) -> int:
        cached = self._direction_index_cache.get(direction)
        if cached is not None:
            return int(cached)
        resolved = DIRECTION_TO_INDEX.get(direction, DIRECTION_TO_INDEX["unknown"])
        self._direction_index_cache[direction] = int(resolved)
        return int(resolved)

    def _grid_size(self, state: Dict[str, Any]) -> Tuple[int, int]:
        raw = state.get("grid_size")
        if (
            isinstance(raw, list)
            and len(raw) == 2
            and isinstance(raw[0], int)
            and isinstance(raw[1], int)
        ):
            return max(1, int(raw[0])), max(1, int(raw[1]))
        return 1, 1

    def _centered_coord(self, value: int, limit: int) -> float:
        span = max(1, int(limit) - 1)
        return (2.0 * float(value) / float(span)) - 1.0

    def _axis_size(self, limit: int) -> float:
        span = max(1, int(limit) - 1)
        return 2.0 / float(span)


class SetAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=~mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class PoolingMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_seeds: int, dropout: float):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        *,
        queries: Optional[Tensor] = None,
    ) -> Tensor:
        if queries is None:
            resolved_queries = self.seed.expand(x.shape[0], -1, -1)
        else:
            resolved_queries = queries
            if resolved_queries.ndim == 2:
                resolved_queries = resolved_queries.unsqueeze(1)
            if resolved_queries.ndim != 3:
                raise ValueError("Pooling queries must have shape [batch, num_seeds, embed_dim].")
            if int(resolved_queries.size(0)) != int(x.shape[0]):
                if int(resolved_queries.size(0)) == 1 and int(x.shape[0]) > 1:
                    resolved_queries = resolved_queries.expand(int(x.shape[0]), -1, -1)
                else:
                    raise ValueError("Pooling query batch size must match encoded token batch size.")
        attn_out, _ = self.attn(
            resolved_queries,
            x,
            x,
            key_padding_mask=~mask,
            need_weights=False,
        )
        pooled = self.norm1(resolved_queries + attn_out)
        pooled = self.norm2(pooled + self.ff(pooled))
        return pooled


class SetStateEncoder(nn.Module):
    """Set Transformer encoder over entity tokens."""

    def __init__(
        self,
        token_dim: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
        num_pool_seeds: int = 1,
    ):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, embed_dim)
        self.blocks = nn.ModuleList(
            SetAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(max(1, int(num_blocks)))
        )
        self.num_pool_seeds = max(1, int(num_pool_seeds))
        self.pool = PoolingMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_seeds=self.num_pool_seeds,
            dropout=dropout,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def encode_context(self, tokens: Tensor, mask: Tensor) -> Tensor:
        x = self.token_proj(tokens)
        for block in self.blocks:
            x = block(x, mask)
        return x

    def pool_context(
        self,
        contextualized_tokens: Tensor,
        mask: Tensor,
        *,
        queries: Optional[Tensor] = None,
    ) -> Tensor:
        pooled = self.pool(contextualized_tokens, mask, queries=queries)
        if int(pooled.size(1)) == 1:
            pooled = pooled.squeeze(1)
        return self.out(pooled)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        contextualized_tokens = self.encode_context(tokens, mask)
        return self.pool_context(contextualized_tokens, mask)
