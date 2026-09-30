from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence
import zipfile

from src.data.state_store import StateStore


@dataclass(frozen=True, slots=True)
class AnalysisArchiveCopyJob:
    src: Path
    dst: Path
    kind: str
    iteration: Optional[int] = None
    bytes_hint: int = 0
    text: Optional[str] = None


@dataclass(frozen=True, slots=True)
class AnalysisArchiveCopyResult:
    iteration: Optional[int]
    bytes_hint: int
    failed: bool


def _analysis_archive_copy_loop(
    job_queue: Any,
    result_queue: Any,
    interval_sec: float,
) -> None:
    interval_sec = max(0.1, float(interval_sec))
    pending_jobs: List[AnalysisArchiveCopyJob] = []
    next_flush_at: Optional[float] = None

    def flush_pending_jobs() -> None:
        nonlocal next_flush_at
        jobs = list(pending_jobs)
        pending_jobs.clear()
        next_flush_at = None
        for job in jobs:
            failed = False
            try:
                AnalysisArchiveCopyWorker.copy_job(job)
            except Exception:
                failed = True
                AnalysisArchiveCopyWorker._remove_copying_destination(job.dst)
            result_queue.put(
                AnalysisArchiveCopyResult(
                    iteration=job.iteration,
                    bytes_hint=max(0, int(job.bytes_hint)),
                    failed=bool(failed),
                )
            )

    while True:
        try:
            if next_flush_at is None:
                first_job = job_queue.get()
            else:
                timeout = max(0.0, float(next_flush_at) - time.monotonic())
                first_job = job_queue.get(timeout=timeout)
        except queue.Empty:
            flush_pending_jobs()
            continue
        if first_job is None:
            flush_pending_jobs()
            return
        else:
            pending_jobs.append(first_job)
            if next_flush_at is None:
                next_flush_at = time.monotonic() + interval_sec
        stop_requested = False
        while True:
            try:
                next_job = job_queue.get_nowait()
            except queue.Empty:
                break
            if next_job is None:
                stop_requested = True
                continue
            pending_jobs.append(next_job)
        if stop_requested or (
            next_flush_at is not None and time.monotonic() >= float(next_flush_at)
        ):
            flush_pending_jobs()
        if stop_requested:
            return


class AnalysisArchiveCopyWorker:
    _ZIP_COMPRESSLEVEL = 1
    _GZIP_COMPRESSLEVEL = 1
    _CLOSE_JOIN_TIMEOUT_SEC = 30.0
    _CLOSE_TERMINATE_TIMEOUT_SEC = 5.0

    def __init__(self, *, interval_sec: float = 5.0) -> None:
        self.interval_sec = max(0.1, float(interval_sec))
        self._ctx = mp.get_context("spawn")
        self._job_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_analysis_archive_copy_loop,
            args=(self._job_queue, self._result_queue, self.interval_sec),
            name="analysis-archive-copy",
            daemon=True,
        )
        self._lock = threading.Lock()
        self._pending_files = 0
        self._pending_bytes = 0
        self._copy_errors = 0
        self._last_copied_iteration: Optional[int] = None
        self._closed = False
        self._process.start()

    def submit_many(self, jobs: Sequence[AnalysisArchiveCopyJob]) -> None:
        if not jobs:
            return
        self._drain_results()
        pending_bytes = sum(max(0, int(job.bytes_hint)) for job in jobs)
        with self._lock:
            self._pending_files += len(jobs)
            self._pending_bytes += int(pending_bytes)
        for job in jobs:
            self._job_queue.put(job)

    def status(self) -> Dict[str, Any]:
        if not self._closed:
            self._drain_results()
        with self._lock:
            return {
                "archive_pending_files": int(self._pending_files),
                "archive_pending_bytes": int(self._pending_bytes),
                "archive_copy_errors": int(self._copy_errors),
                "archive_last_copied_iteration": self._last_copied_iteration,
            }

    def close(self) -> None:
        if self._closed:
            return
        timed_out = False
        self._job_queue.put(None)
        self._process.join(timeout=float(self._CLOSE_JOIN_TIMEOUT_SEC))
        if self._process.is_alive():
            timed_out = True
            self._process.terminate()
            self._process.join(timeout=float(self._CLOSE_TERMINATE_TIMEOUT_SEC))
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=1.0)
        self._drain_results()
        self._closed = True
        if timed_out:
            self._job_queue.cancel_join_thread()
            self._result_queue.cancel_join_thread()
        self._job_queue.close()
        if not timed_out:
            self._job_queue.join_thread()
        self._result_queue.close()
        if not timed_out:
            self._result_queue.join_thread()

    @classmethod
    def copy_job(cls, job: AnalysisArchiveCopyJob) -> None:
        try:
            if job.kind == "dir":
                cls._copy_dir(job.src, job.dst)
            elif job.kind == "dir_zip":
                cls._copy_dir_as_zip(job.src, job.dst)
            elif job.kind == "file":
                cls._copy_file(job.src, job.dst)
            elif job.kind == "file_gzip":
                cls._copy_file_as_gzip(job.src, job.dst)
            elif job.kind == "text":
                cls._copy_text(str(job.text or ""), job.dst)
            else:
                raise ValueError(f"Unknown analysis archive copy job kind: {job.kind!r}")
        except Exception:
            cls._remove_copying_destination(job.dst)
            raise

    def _drain_results(self) -> None:
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                return
            self._mark_job_done(result)

    def _mark_job_done(self, result: AnalysisArchiveCopyResult) -> None:
        with self._lock:
            self._pending_files = max(0, self._pending_files - 1)
            self._pending_bytes = max(
                0,
                self._pending_bytes - max(0, int(result.bytes_hint)),
            )
            if bool(result.failed):
                self._copy_errors += 1
            elif result.iteration is not None:
                copied_iteration = int(result.iteration)
                if (
                    self._last_copied_iteration is None
                    or copied_iteration > int(self._last_copied_iteration)
                ):
                    self._last_copied_iteration = copied_iteration

    @staticmethod
    def _copy_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_name(dst.name + ".copying")
        AnalysisArchiveCopyWorker._remove_path(tmp_dst)
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)

    @classmethod
    def _copy_file_as_gzip(cls, src: Path, dst: Path) -> None:
        if not src.is_file():
            raise FileNotFoundError(f"Archive source file not found: {src}")
        with tempfile.TemporaryDirectory(
            prefix=f"{src.stem}_gzip_",
            dir=str(src.parent),
        ) as tmpdir:
            archive_path = Path(tmpdir) / f"{src.name}.gz"
            cls._write_file_gzip(src, archive_path)
            cls._copy_file(archive_path, dst)

    @classmethod
    def _write_file_gzip(cls, src: Path, archive_path: Path) -> None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as source, gzip.open(
            archive_path,
            "wb",
            compresslevel=cls._GZIP_COMPRESSLEVEL,
        ) as handle:
            shutil.copyfileobj(source, handle)

    @staticmethod
    def _copy_text(text: str, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_name(dst.name + ".copying")
        AnalysisArchiveCopyWorker._remove_path(tmp_dst)
        tmp_dst.write_text(text, encoding="utf-8")
        os.replace(tmp_dst, dst)

    @staticmethod
    def _copy_dir(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_name(dst.name + ".copying")
        AnalysisArchiveCopyWorker._remove_path(tmp_dst)
        if dst.exists():
            return
        shutil.copytree(src, tmp_dst)
        os.rename(tmp_dst, dst)

    @classmethod
    def _copy_dir_as_zip(cls, src: Path, dst: Path) -> None:
        if not src.is_dir():
            raise FileNotFoundError(f"State-store segment directory not found: {src}")
        with tempfile.TemporaryDirectory(
            prefix=f"{src.name}_zip_",
            dir=str(src.parent),
        ) as tmpdir:
            archive_path = Path(tmpdir) / f"{src.name}.zip"
            cls._write_dir_zip(src, archive_path)
            cls._copy_file(archive_path, dst)

    @classmethod
    def _write_dir_zip(cls, src: Path, archive_path: Path) -> None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        root_name = src.name
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=cls._ZIP_COMPRESSLEVEL,
        ) as handle:
            for child in sorted(src.rglob("*")):
                if not child.is_file():
                    continue
                relative_child = child.relative_to(src).as_posix()
                archive_name = f"{root_name}/{relative_child}"
                handle.write(child, arcname=archive_name)

    @staticmethod
    def _remove_copying_destination(dst: Path) -> None:
        tmp_dst = dst.with_name(dst.name + ".copying")
        AnalysisArchiveCopyWorker._remove_path(tmp_dst)

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


class AnalysisArchiveWriter:
    def __init__(
        self,
        *,
        exp_root: str | Path,
        local_root: str | Path | None = None,
        copy_interval_sec: float = 5.0,
    ) -> None:
        self.exp_root = Path(exp_root).absolute()
        if local_root is None:
            local_root = (
                Path(tempfile.gettempdir())
                / "baba_analysis_archive"
                / self.exp_root.parent.name
                / self.exp_root.name
        )
        self.local_root = Path(local_root).resolve()
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.exp_root.mkdir(parents=True, exist_ok=True)
        self._copy_worker = AnalysisArchiveCopyWorker(
            interval_sec=float(copy_interval_sec)
        )
        self._last_saved_state_id = 0
        self._state_segments: List[Dict[str, Any]] = []

    def close(self) -> None:
        self._copy_worker.close()

    def status(self) -> Dict[str, Any]:
        return self._copy_worker.status()

    def record_iteration(
        self,
        *,
        iteration: int,
        metric_rows: Sequence[Mapping[str, Any]],
        iteration_summary: Mapping[str, Any],
        node_rows: Sequence[Mapping[str, Any]],
        edge_rows: Sequence[Mapping[str, Any]],
        state_store: Optional[StateStore],
    ) -> Dict[str, Any]:
        jobs: List[AnalysisArchiveCopyJob] = []
        metric_path = self._write_jsonl_raw(
            "metrics",
            iteration=iteration,
            rows=self._metric_archive_rows(
                iteration=iteration,
                metric_rows=metric_rows,
                iteration_summary=iteration_summary,
            ),
        )
        if metric_path is not None:
            jobs.append(
                self._copy_job(
                    metric_path,
                    kind="file_gzip",
                    iteration=iteration,
                    dst_relative=self._jsonl_gzip_relative("metrics", iteration),
                )
            )

        nodes_path = self._write_jsonl_raw(
            "nodes",
            iteration=iteration,
            rows=node_rows,
        )
        if nodes_path is not None:
            jobs.append(
                self._copy_job(
                    nodes_path,
                    kind="file_gzip",
                    iteration=iteration,
                    dst_relative=self._jsonl_gzip_relative("nodes", iteration),
                )
            )

        edges_path = self._write_jsonl_raw(
            "edges",
            iteration=iteration,
            rows=edge_rows,
        )
        if edges_path is not None:
            jobs.append(
                self._copy_job(
                    edges_path,
                    kind="file_gzip",
                    iteration=iteration,
                    dst_relative=self._jsonl_gzip_relative("edges", iteration),
                )
            )

        jobs.extend(
            self._write_state_store_segment_jobs(
                iteration=iteration,
                state_store=state_store,
            )
        )
        self._copy_worker.submit_many(jobs)
        return self.status()

    def _write_jsonl_raw(
        self,
        dirname: str,
        *,
        iteration: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> Optional[Path]:
        if not rows:
            return None
        output_dir = self.local_root / dirname
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"iter_{int(iteration):06d}.jsonl"
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
        os.replace(tmp_path, path)
        return path

    @staticmethod
    def _jsonl_gzip_relative(dirname: str, iteration: int) -> Path:
        return Path(dirname) / f"iter_{int(iteration):06d}.jsonl.gz"

    def _metric_archive_rows(
        self,
        *,
        iteration: int,
        metric_rows: Sequence[Mapping[str, Any]],
        iteration_summary: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        start_global_step: Optional[int] = None
        end_global_step: Optional[int] = None
        for row in metric_rows:
            archived = dict(row)
            archived["row_type"] = "step"
            archived["iteration"] = int(iteration)
            raw_global_step = archived.get("global_step")
            if isinstance(raw_global_step, int):
                if start_global_step is None:
                    start_global_step = int(raw_global_step)
                end_global_step = int(raw_global_step)
            rows.append(archived)
        summary = {
            "row_type": "iteration_summary",
            "iteration": int(iteration),
            "start_global_step": start_global_step,
            "end_global_step": end_global_step,
            "step_count": int(len(metric_rows)),
            "global_step": iteration_summary.get("global_step"),
            "iteration_end_program_version": iteration_summary.get("program_version"),
            "llm_calls": iteration_summary.get("llm_calls"),
            "archive_local_root": str(self.local_root),
            "archive_exp_root": str(self.exp_root),
        }
        summary.update(self.status())
        rows.append(summary)
        return rows

    def _write_state_store_segment_jobs(
        self,
        *,
        iteration: int,
        state_store: Optional[StateStore],
    ) -> List[AnalysisArchiveCopyJob]:
        if not isinstance(state_store, StateStore):
            return []
        current_state_id = int(len(state_store))
        first_state_id = int(self._last_saved_state_id) + 1
        if current_state_id < first_state_id:
            return []

        state_root = self.local_root / "state_store"
        segment_name = f"{first_state_id:09d}_{current_state_id:09d}"
        segment_dir = state_root / "segments" / segment_name
        state_store.save_directory_slice(
            segment_dir,
            first_state_id=first_state_id,
            last_state_id=current_state_id,
        )
        self._last_saved_state_id = current_state_id
        self._state_segments.append(
            {
                "firstStateId": int(first_state_id),
                "lastStateId": int(current_state_id),
                "segmentName": segment_name,
            }
        )
        manifest_path = self._write_state_store_manifest(
            state_root,
            published=False,
        )
        manifest_text = self._state_store_manifest_text(published=True)
        return [
            self._copy_job(
                segment_dir,
                kind="dir_zip",
                iteration=iteration,
                dst_relative=Path("state_store") / "segments" / f"{segment_name}.zip",
            ),
            AnalysisArchiveCopyJob(
                src=manifest_path.resolve(),
                dst=(self.exp_root / "state_store" / "segments.json").resolve(),
                kind="text",
                iteration=int(iteration),
                bytes_hint=len(manifest_text.encode("utf-8")),
                text=manifest_text,
            ),
        ]

    def _write_state_store_manifest(
        self,
        state_root: Path,
        *,
        published: bool,
    ) -> Path:
        state_root.mkdir(parents=True, exist_ok=True)
        manifest_path = state_root / "segments.json"
        tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
        manifest_text = self._state_store_manifest_text(published=published)
        tmp_path.write_text(manifest_text, encoding="utf-8")
        os.replace(tmp_path, manifest_path)
        return manifest_path

    def _state_store_manifest_text(self, *, published: bool) -> str:
        return json.dumps(
            self._state_store_manifest_payload(published=published),
            ensure_ascii=False,
            sort_keys=True,
        )

    def _state_store_manifest_payload(self, *, published: bool) -> Dict[str, Any]:
        segments: List[Dict[str, Any]] = []
        for segment in self._state_segments:
            segment_name = str(segment.get("segmentName", "")).strip()
            if not segment_name:
                continue
            suffix = ".zip" if published else ""
            storage = "zip" if published else "directory"
            segments.append(
                {
                    "firstStateId": int(segment.get("firstStateId", 0) or 0),
                    "lastStateId": int(segment.get("lastStateId", 0) or 0),
                    "path": f"segments/{segment_name}{suffix}",
                    "storage": storage,
                }
            )
        return {
            "formatVersion": 2,
            "stateCount": int(self._last_saved_state_id),
            "segments": segments,
        }

    def _copy_job(
        self,
        path: Path,
        *,
        kind: str,
        iteration: int,
        dst_relative: Optional[str | Path] = None,
    ) -> AnalysisArchiveCopyJob:
        if dst_relative is None:
            relative = path.resolve().relative_to(self.local_root)
        else:
            relative = Path(dst_relative)
        return AnalysisArchiveCopyJob(
            src=path.resolve(),
            dst=(self.exp_root / relative).absolute(),
            kind=str(kind),
            iteration=int(iteration),
            bytes_hint=self._path_size(path),
        )

    @classmethod
    def _path_size(cls, path: Path) -> int:
        if path.is_file():
            return int(path.stat().st_size)
        total = 0
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    total += int(child.stat().st_size)
        return int(total)
