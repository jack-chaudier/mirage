from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from narrativefield.storyteller.types import NarrativeStateObject


logger = logging.getLogger(__name__)


def _write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically write text to `path` (POSIX rename semantics)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.debug("Failed to clean up temp checkpoint file %s", tmp_path, exc_info=True)
        raise


class CheckpointManager:
    """Saves and restores pipeline state for crash recovery.

    After each successful scene generation:
    1. Serialize NarrativeStateObject to JSON
    2. Save all generated prose chunks
    3. Record which scene we're on

    On resume:
    1. Load the last successful checkpoint
    2. Continue from the next scene (i.e., last_completed_scene + 1)
    """

    def __init__(self, checkpoint_dir: str, run_id: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_id = run_id
        self.run_dir = self.checkpoint_dir / run_id

    def save(self, state: NarrativeStateObject, prose_chunks: list[str], scene_index: int) -> None:
        """Save checkpoint after successful scene generation."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        scene_idx = int(scene_index)

        state_path = self.run_dir / f"state_scene_{scene_idx:02d}.json"
        prose_dir = self.run_dir

        try:
            state_payload = state.to_dict()
            _write_text_atomic(
                state_path,
                json.dumps(state_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Write/refresh prose chunks so the checkpoint is self-contained.
            for i, chunk in enumerate(prose_chunks):
                _write_text_atomic(prose_dir / f"prose_scene_{i:02d}.txt", str(chunk), encoding="utf-8")

            manifest = {
                "run_id": self.run_id,
                "last_completed_scene": scene_idx,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            _write_text_atomic(
                self.run_dir / "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.exception("Failed to save checkpoint run_id=%s scene=%s: %s", self.run_id, scene_idx, e)
            raise

    def load_latest(self) -> tuple[NarrativeStateObject, list[str], int] | None:
        """Load most recent checkpoint, or None if no checkpoint exists.

        Returns (state, prose_chunks, last_completed_scene).
        """

        if not self.has_checkpoint():
            return None

        try:
            manifest_path = self.run_dir / "manifest.json"
            last_completed_scene: int | None = None
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest, dict):
                    last_completed_scene = int(manifest.get("last_completed_scene", 0) or 0)

            if last_completed_scene is None:
                last_completed_scene = self._infer_last_completed_scene()
                if last_completed_scene is None:
                    return None

            state_path = self.run_dir / f"state_scene_{last_completed_scene:02d}.json"
            if not state_path.exists():
                return None
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state_data, dict):
                return None
            state = NarrativeStateObject.from_dict(state_data)

            prose_chunks: list[str] = []
            for i in range(last_completed_scene + 1):
                p = self.run_dir / f"prose_scene_{i:02d}.txt"
                if p.exists():
                    prose_chunks.append(p.read_text(encoding="utf-8"))
                else:
                    prose_chunks.append("")

            return (state, prose_chunks, last_completed_scene)
        except Exception as e:
            logger.exception("Failed to load checkpoint run_id=%s: %s", self.run_id, e)
            return None

    def _infer_last_completed_scene(self) -> int | None:
        candidates: list[int] = []
        for p in self.run_dir.glob("state_scene_*.json"):
            stem = p.stem  # state_scene_XX
            try:
                idx = int(stem.split("_")[-1])
                candidates.append(idx)
            except Exception:
                continue
        return max(candidates) if candidates else None

    def has_checkpoint(self) -> bool:
        if not self.run_dir.exists():
            return False
        if (self.run_dir / "manifest.json").exists():
            return True
        return any(self.run_dir.glob("state_scene_*.json"))

    def clear(self) -> None:
        """Remove all checkpoints for this run (call after successful completion)."""

        if self.run_dir.exists():
            shutil.rmtree(self.run_dir, ignore_errors=True)
