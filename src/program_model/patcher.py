import ast
import hashlib
import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .contract import PREDICT_FUNCTION_NAME


@dataclass
class PatchApplyResult:
    success: bool
    updated_source: Optional[str] = None
    error_message: Optional[str] = None
    normalized_patch: Optional[Any] = None
    patch_digest: Optional[str] = None


@dataclass
class _PatchHunk:
    old_lines: List[str]
    new_lines: List[str]


@dataclass(frozen=True)
class _FunctionTarget:
    name: str
    path: tuple[tuple[str, int], ...]
    is_nested: bool


class ProgramPatcher:
    """Apply either line-based or function-based patches to a Python source string."""

    TARGET_FILENAME = "accepted_baseline_program.py"
    SUPPORTED_PATCH_FORMATS = ("line", "function")
    _ALLOWED_FUNCTION_OPS = {"replace", "add"}

    def __init__(self, patch_format: str = "line") -> None:
        self.patch_format = self.normalize_patch_format(patch_format)

    @classmethod
    def normalize_patch_format(cls, patch_format: Any) -> str:
        normalized = "line" if patch_format is None else str(patch_format).strip().lower()
        if normalized not in cls.SUPPORTED_PATCH_FORMATS:
            allowed = ", ".join(cls.SUPPORTED_PATCH_FORMATS)
            raise ValueError(
                f"Unsupported patch_format={patch_format!r}. Allowed values: {allowed}."
            )
        return normalized

    @property
    def prompt_template_name(self) -> str:
        if self.patch_format == "function":
            return "patch_generation_function.txt"
        return "patch_generation.txt"

    def normalize_patch_payload(self, payload: Any) -> Any:
        if self.patch_format == "function":
            return self._normalize_function_patch_payload(payload)
        normalized_patch, _ = self._parse_and_normalize_line_patch(payload)
        return normalized_patch

    def apply_patch(self, base_source: str, patch_payload: Any) -> PatchApplyResult:
        if self.patch_format == "function":
            return self._apply_function_patch(base_source, patch_payload)
        return self._apply_line_patch(base_source, patch_payload)

    def _apply_function_patch(self, base_source: str, patch_payload: Any) -> PatchApplyResult:
        try:
            normalized = self._normalize_function_patch_payload(patch_payload)
        except ValueError as e:
            return PatchApplyResult(success=False, error_message=str(e))

        try:
            module = ast.parse(base_source)
        except SyntaxError as e:
            return PatchApplyResult(
                success=False,
                error_message=f"Base source has syntax error: {e}",
            )

        for name in normalized["delete_functions"]:
            try:
                target = self._resolve_function_target(
                    self._build_function_target_index(module),
                    name=name,
                    prefer_nested=False,
                )
            except ValueError as e:
                return PatchApplyResult(success=False, error_message=str(e))
            if target is None:
                continue
            self._delete_node_at_path(module, target.path)

        for edit in normalized["edits"]:
            op = edit["op"]
            name = edit["name"]
            parsed_fn = self._parse_function_snippet(edit["code"], expected_name=name)
            if parsed_fn is None:
                return PatchApplyResult(
                    success=False,
                    error_message=f"Invalid function snippet for `{name}`.",
                )

            if op == "replace":
                function_targets = self._build_function_target_index(module)
                prefer_nested = self._snippet_prefers_nested_target(edit["code"])
                try:
                    existing_target = self._resolve_function_target(
                        function_targets,
                        name=name,
                        prefer_nested=prefer_nested,
                    )
                except ValueError as e:
                    return PatchApplyResult(success=False, error_message=str(e))
                if existing_target is None:
                    return PatchApplyResult(
                        success=False,
                        error_message=f"replace target `{name}` does not exist.",
                    )
                self._replace_node_at_path(module, existing_target.path, parsed_fn)
            else:
                function_targets = self._build_function_target_index(module)
                if function_targets.get(name):
                    return PatchApplyResult(
                        success=False,
                        error_message=(
                            f"add target `{name}` already exists. "
                            "Use replace for existing functions."
                        ),
                    )
                module.body.append(parsed_fn)

        if PREDICT_FUNCTION_NAME not in self._build_function_positions(list(module.body)):
            return PatchApplyResult(
                success=False,
                error_message="Patched source must define predict_next_state.",
            )

        ast.fix_missing_locations(module)
        try:
            updated_source = ast.unparse(module).rstrip() + "\n"
            compile(updated_source, "<patched_program>", "exec")
        except Exception as e:  # noqa: BLE001
            return PatchApplyResult(
                success=False,
                error_message=f"Patched source failed syntax/compile validation: {e}",
            )

        patch_json = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(patch_json.encode("utf-8")).hexdigest()
        return PatchApplyResult(
            success=True,
            updated_source=updated_source,
            normalized_patch=normalized,
            patch_digest=digest,
        )

    def _apply_line_patch(self, base_source: str, patch_payload: Any) -> PatchApplyResult:
        try:
            normalized_patch, hunks = self._parse_and_normalize_line_patch(patch_payload)
        except ValueError as e:
            return PatchApplyResult(success=False, error_message=str(e))

        try:
            updated_source = self._apply_hunks(base_source, hunks)
        except ValueError as e:
            return PatchApplyResult(
                success=False,
                error_message=str(e),
            )

        try:
            parsed_module = ast.parse(updated_source)
        except SyntaxError as e:
            return PatchApplyResult(
                success=False,
                error_message=f"Patched source failed syntax validation: {e}",
            )

        if not any(
            getattr(node, "name", None) == PREDICT_FUNCTION_NAME
            for node in parsed_module.body
        ):
            return PatchApplyResult(
                success=False,
                error_message="Patched source must define predict_next_state.",
            )

        try:
            compile(updated_source, "<patched_program>", "exec")
        except Exception as e:  # noqa: BLE001
            return PatchApplyResult(
                success=False,
                error_message=f"Patched source failed syntax/compile validation: {e}",
            )

        digest = hashlib.sha256(normalized_patch.encode("utf-8")).hexdigest()
        return PatchApplyResult(
            success=True,
            updated_source=updated_source,
            normalized_patch=normalized_patch,
            patch_digest=digest,
        )

    def _normalize_function_patch_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, str):
            obj_text = self._extract_function_patch_json_object(payload)
            if obj_text is None:
                raise ValueError("Patch JSON object not found in model output.")
            try:
                payload = json.loads(obj_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid patch JSON: {e}") from e

        if not isinstance(payload, dict):
            raise ValueError("Patch payload must be a JSON object.")

        edits = payload.get("edits")
        delete_functions = payload.get("delete_functions", [])

        if not isinstance(edits, list):
            raise ValueError("Patch payload must contain list field `edits`.")
        if not isinstance(delete_functions, list):
            raise ValueError("`delete_functions` must be a list.")

        norm_edits: List[Dict[str, str]] = []
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                raise ValueError(f"edits[{i}] must be an object.")
            op = edit.get("op")
            name = edit.get("name")
            code = edit.get("code")
            if op not in self._ALLOWED_FUNCTION_OPS:
                raise ValueError(
                    f"edits[{i}].op must be one of {sorted(self._ALLOWED_FUNCTION_OPS)}."
                )
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"edits[{i}].name must be a non-empty string.")
            if not isinstance(code, str) or not code.strip():
                raise ValueError(f"edits[{i}].code must be a non-empty string.")
            cleaned_name = name.strip()
            if op == "add":
                if cleaned_name == PREDICT_FUNCTION_NAME:
                    raise ValueError(
                        "add cannot target predict_next_state; "
                        "use replace instead."
                    )
                if self._snippet_prefers_nested_target(code):
                    raise ValueError(
                        "add code must define an unindented top-level "
                        "function. Use replace for existing helpers."
                    )
            norm_edits.append({"op": op, "name": cleaned_name, "code": code})

        norm_delete: List[str] = []
        for i, name in enumerate(delete_functions):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"delete_functions[{i}] must be a non-empty function name."
                )
            cleaned = name.strip()
            if cleaned == PREDICT_FUNCTION_NAME:
                raise ValueError("predict_next_state cannot be deleted.")
            norm_delete.append(cleaned)

        return {"edits": norm_edits, "delete_functions": norm_delete}

    def _parse_and_normalize_line_patch(self, payload: Any) -> tuple[str, List[_PatchHunk]]:
        patch_text = self._coerce_line_patch_text(payload)
        lines = patch_text.splitlines()
        if not lines or lines[0].strip() != "*** Begin Patch":
            raise ValueError("Patch must start with `*** Begin Patch`.")
        if lines[-1].strip() != "*** End Patch":
            raise ValueError("Patch must end with `*** End Patch`.")

        normalized_lines = ["*** Begin Patch"]
        hunks: List[_PatchHunk] = []
        update_seen = False
        i = 1

        while i < len(lines) - 1:
            line = lines[i]
            if line.startswith("*** Add File: ") or line.startswith("*** Delete File: "):
                raise ValueError("Only `*** Update File:` patches are supported.")
            if line.startswith("*** Move to: "):
                raise ValueError("File moves are not supported.")
            if line == "*** End of File":
                i += 1
                continue
            if not line.startswith("*** Update File: "):
                raise ValueError(
                    "Patch body must contain a single `*** Update File:` section."
                )
            if update_seen:
                raise ValueError("Only one `*** Update File:` section is supported.")

            update_seen = True
            normalized_lines.append(f"*** Update File: {self.TARGET_FILENAME}")
            i += 1
            hunk_count = 0

            while i < len(lines) - 1:
                line = lines[i]
                if line.startswith("*** "):
                    break
                if not line.startswith("@@"):
                    raise ValueError("Each patch block must start with `@@`.")
                normalized_lines.append("@@")
                i += 1
                hunk_lines: List[tuple[str, str]] = []
                while i < len(lines) - 1:
                    line = lines[i]
                    if line.startswith("@@") or line.startswith("*** "):
                        break
                    if not line:
                        raise ValueError(
                            "Hunk lines must start with a space, `+`, or `-`."
                        )
                    prefix = line[0]
                    if prefix not in {" ", "+", "-"}:
                        raise ValueError(
                            "Hunk lines must start with a space, `+`, or `-`."
                        )
                    text = line[1:]
                    hunk_lines.append((prefix, text))
                    normalized_lines.append(prefix + text)
                    i += 1
                hunks.append(self._build_hunk(hunk_lines))
                hunk_count += 1

            if hunk_count == 0:
                raise ValueError("Patch must contain at least one hunk.")
            continue

        if not update_seen:
            raise ValueError("Patch must contain one `*** Update File:` section.")

        normalized_lines.append("*** End Patch")
        return "\n".join(normalized_lines) + "\n", hunks

    def _build_function_positions(self, body: List[ast.stmt]) -> Dict[str, int]:
        positions: Dict[str, int] = {}
        for i, node in enumerate(body):
            if isinstance(node, ast.FunctionDef):
                positions[node.name] = i
        return positions

    def _build_function_target_index(
        self,
        node: ast.AST,
        *,
        path: tuple[tuple[str, int], ...] = (),
        function_depth: int = 0,
    ) -> Dict[str, List[_FunctionTarget]]:
        targets: Dict[str, List[_FunctionTarget]] = {}
        for field_name, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for idx, item in enumerate(value):
                if not isinstance(item, ast.AST):
                    continue
                item_path = path + ((field_name, idx),)
                if isinstance(item, ast.FunctionDef):
                    target = _FunctionTarget(
                        name=item.name,
                        path=item_path,
                        is_nested=function_depth > 0,
                    )
                    targets.setdefault(item.name, []).append(target)
                    child_targets = self._build_function_target_index(
                        item,
                        path=item_path,
                        function_depth=function_depth + 1,
                    )
                else:
                    child_targets = self._build_function_target_index(
                        item,
                        path=item_path,
                        function_depth=function_depth,
                    )
                for name, entries in child_targets.items():
                    targets.setdefault(name, []).extend(entries)
        return targets

    def _resolve_function_target(
        self,
        function_targets: Dict[str, List[_FunctionTarget]],
        *,
        name: str,
        prefer_nested: bool,
    ) -> Optional[_FunctionTarget]:
        matches = function_targets.get(name, [])
        if not matches:
            return None

        top_level = [target for target in matches if not target.is_nested]
        nested = [target for target in matches if target.is_nested]
        ordered_groups = [nested, top_level] if prefer_nested else [top_level, nested]

        for group in ordered_groups:
            if not group:
                continue
            if len(group) > 1:
                scope = "nested " if group[0].is_nested else ""
                raise ValueError(f"{scope}function target `{name}` is ambiguous.")
            return group[0]
        return None

    def _replace_node_at_path(
        self,
        root: ast.AST,
        path: tuple[tuple[str, int], ...],
        new_node: ast.AST,
    ) -> None:
        parent = root
        for field_name, idx in path[:-1]:
            parent = getattr(parent, field_name)[idx]
        field_name, idx = path[-1]
        getattr(parent, field_name)[idx] = new_node

    def _delete_node_at_path(
        self,
        root: ast.AST,
        path: tuple[tuple[str, int], ...],
    ) -> None:
        parent = root
        for field_name, idx in path[:-1]:
            parent = getattr(parent, field_name)[idx]
        field_name, idx = path[-1]
        del getattr(parent, field_name)[idx]

    def _parse_function_snippet(
        self,
        code: str,
        expected_name: Optional[str] = None,
    ) -> Optional[ast.FunctionDef]:
        fn = self._parse_top_level_function_snippet(code)
        if fn is None:
            fn = self._parse_nested_function_snippet(code)
        if fn is None:
            return None
        if expected_name is not None and fn.name != expected_name:
            return None
        return fn

    def _parse_top_level_function_snippet(self, code: str) -> Optional[ast.FunctionDef]:
        try:
            snippet_tree = ast.parse(code)
        except SyntaxError:
            return None
        funcs = [node for node in snippet_tree.body if isinstance(node, ast.FunctionDef)]
        if len(funcs) != 1:
            return None
        return funcs[0]

    def _parse_nested_function_snippet(self, code: str) -> Optional[ast.FunctionDef]:
        if not self._snippet_prefers_nested_target(code):
            return None
        snippet_body = code.strip("\n")
        if not snippet_body:
            return None
        wrapped_code = "def __patch_wrapper__():\n" + textwrap.indent(snippet_body, "    ")
        try:
            snippet_tree = ast.parse(wrapped_code)
        except SyntaxError:
            return None
        wrapper = snippet_tree.body[0]
        if not isinstance(wrapper, ast.FunctionDef):
            return None
        funcs = [node for node in wrapper.body if isinstance(node, ast.FunctionDef)]
        if len(funcs) != 1:
            return None
        return funcs[0]

    def _snippet_prefers_nested_target(self, code: str) -> bool:
        for line in code.splitlines():
            if not line.strip():
                continue
            return line[:1].isspace()
        return False

    def _build_hunk(self, hunk_lines: List[tuple[str, str]]) -> _PatchHunk:
        if not hunk_lines:
            raise ValueError("Patch hunk must contain at least one line.")

        old_lines = [text for prefix, text in hunk_lines if prefix in {" ", "-"}]
        new_lines = [text for prefix, text in hunk_lines if prefix in {" ", "+"}]
        has_change = any(prefix in {"+", "-"} for prefix, _ in hunk_lines)

        if not has_change:
            raise ValueError("Patch hunk must include at least one added or removed line.")
        if not old_lines:
            raise ValueError(
                "Patch hunk must include at least one context or removed line to anchor it."
            )

        return _PatchHunk(old_lines=old_lines, new_lines=new_lines)

    def _coerce_line_patch_text(self, payload: Any) -> str:
        if isinstance(payload, dict):
            raw_text = payload.get("patch")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError("Patch payload object must contain a non-empty `patch` string.")
            decoded_text = raw_text.strip()
        elif isinstance(payload, str):
            raw_text = payload.strip()
            if not raw_text:
                raise ValueError("Patch payload must not be empty.")
            decoded_text = self._maybe_decode_json_value(raw_text)
        else:
            raise ValueError(
                "Patch payload must be an apply_patch string or an object with a `patch` field."
            )

        extracted = self._extract_patch_block(decoded_text)
        if extracted is None:
            raise ValueError("apply_patch block not found in model output.")
        return extracted

    def _maybe_decode_json_value(self, text: str) -> str:
        if not text.startswith(('"', "{")):
            return text
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(decoded, str):
            return decoded
        if isinstance(decoded, dict):
            patch_text = decoded.get("patch")
            if isinstance(patch_text, str):
                return patch_text
        return text

    def _extract_patch_block(self, text: str) -> Optional[str]:
        start = text.find("*** Begin Patch")
        if start == -1:
            return None
        end = text.find("*** End Patch", start)
        if end == -1:
            return None
        end += len("*** End Patch")
        return text[start:end]

    def _extract_function_patch_json_object(self, text: str) -> Optional[str]:
        selected: Optional[str] = None
        for obj_text in self._iter_json_objects(text):
            try:
                payload = json.loads(obj_text)
            except json.JSONDecodeError:
                continue
            if self._looks_like_function_patch_payload(payload):
                selected = obj_text
        return selected

    def _looks_like_function_patch_payload(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        edits = payload.get("edits")
        if not isinstance(edits, list):
            return False
        delete_functions = payload.get("delete_functions", [])
        return isinstance(delete_functions, list)

    def _iter_json_objects(self, text: str) -> List[str]:
        objects: List[str] = []
        search_start = 0
        while True:
            start = text.find("{", search_start)
            if start == -1:
                return objects

            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue
                if ch == "{":
                    depth += 1
                    continue
                if ch == "}":
                    depth -= 1
                    if depth == 0:
                        objects.append(text[start : i + 1])
                        break

            search_start = start + 1

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _apply_hunks(self, base_source: str, hunks: List[_PatchHunk]) -> str:
        current_lines = base_source.split("\n")
        search_start = 0

        for hunk in hunks:
            match_index = self._find_hunk_start(
                current_lines=current_lines,
                old_lines=hunk.old_lines,
                search_start=search_start,
            )
            current_lines[match_index : match_index + len(hunk.old_lines)] = hunk.new_lines
            search_start = match_index + len(hunk.new_lines)

        return "\n".join(current_lines)

    def _find_hunk_start(
        self,
        *,
        current_lines: List[str],
        old_lines: List[str],
        search_start: int,
    ) -> int:
        forward_matches = self._find_matching_offsets(
            current_lines=current_lines,
            old_lines=old_lines,
            start_index=search_start,
        )
        if len(forward_matches) == 1:
            return forward_matches[0]
        if len(forward_matches) > 1:
            raise ValueError("Patch hunk is ambiguous; add more surrounding context.")

        all_matches = self._find_matching_offsets(
            current_lines=current_lines,
            old_lines=old_lines,
            start_index=0,
        )
        if len(all_matches) == 1:
            return all_matches[0]
        if len(all_matches) > 1:
            raise ValueError("Patch hunk is ambiguous; add more surrounding context.")
        raise ValueError("Patch hunk did not match the current source.")

    def _find_matching_offsets(
        self,
        *,
        current_lines: List[str],
        old_lines: List[str],
        start_index: int,
    ) -> List[int]:
        max_offset = len(current_lines) - len(old_lines)
        if max_offset < start_index:
            return []

        matches: List[int] = []
        for offset in range(start_index, max_offset + 1):
            if current_lines[offset : offset + len(old_lines)] == old_lines:
                matches.append(offset)
        return matches
