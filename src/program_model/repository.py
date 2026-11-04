import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .contract import BASELINE_PROGRAM_SOURCE


@dataclass
class ProgramVersion:
    index: int
    version_id: str
    filepath: Path
    source: str
    parent_version_id: Optional[str]
    created_at: str


class ProgramRepository:
    """Persist accepted program versions and candidate metadata logs."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir = self.root_dir / "program_versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.history_jsonl = self.root_dir / "program_history.jsonl"
        self._versions_cache: Optional[List[ProgramVersion]] = None

    def ensure_baseline(self) -> ProgramVersion:
        versions = self.list_versions()
        if versions:
            return versions[-1]
        return self.save_version(
            source=BASELINE_PROGRAM_SOURCE,
            parent_version_id=None,
            metadata={"accepted": True, "reason": "baseline"},
        )

    def list_versions(self) -> List[ProgramVersion]:
        cached_versions = self._versions_cache
        if cached_versions is not None:
            return list(cached_versions)

        versions: List[ProgramVersion] = []
        for path in sorted(self.versions_dir.glob("v*.py")):
            stem = path.stem
            if not stem.startswith("v") or not stem[1:].isdigit():
                continue
            index = int(stem[1:])
            source = path.read_text(encoding="utf-8")
            versions.append(
                ProgramVersion(
                    index=index,
                    version_id=stem,
                    filepath=path,
                    source=source,
                    parent_version_id=None,
                    created_at=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                )
            )
        self._versions_cache = list(versions)
        return versions

    def save_version(
        self,
        source: str,
        parent_version_id: Optional[str],
        metadata: Optional[Dict] = None,
    ) -> ProgramVersion:
        next_index = self._next_version_index()
        version_id = f"v{next_index:03d}"
        filepath = self.versions_dir / f"{version_id}.py"
        filepath.write_text(source, encoding="utf-8")

        created_at = datetime.now().isoformat()
        entry = {
            "timestamp": created_at,
            "version_id": version_id,
            "parent_version_id": parent_version_id,
            "accepted": True,
        }
        if metadata:
            entry.update(metadata)
        self.record_attempt(entry)

        version = ProgramVersion(
            index=next_index,
            version_id=version_id,
            filepath=filepath,
            source=source,
            parent_version_id=parent_version_id,
            created_at=created_at,
        )
        cached_versions = self._versions_cache
        if cached_versions is None:
            self._versions_cache = [version]
        else:
            cached_versions.append(version)
        return version

    def _next_version_index(self) -> int:
        cached_versions = self._versions_cache
        if cached_versions:
            return int(cached_versions[-1].index) + 1

        max_index = -1
        for path in self.versions_dir.glob("v*.py"):
            stem = path.stem
            if not stem.startswith("v") or not stem[1:].isdigit():
                continue
            index = int(stem[1:])
            if index > max_index:
                max_index = index
        return max_index + 1

    def record_attempt(self, entry: Dict) -> None:
        self.history_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
