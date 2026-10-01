"""Small JSON-valued persistent knowledge store, isolated from shell workspaces."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import tempfile
import threading
from typing import Any

from .protocol import canonical_json


class KnowledgeStore:
    """Bounded, atomic-file KV store for a single agent process.

    Reset or use a fresh path per evaluation episode to prevent test leakage.
    Use an external database/lock for multiple writer processes.
    """
    def __init__(self, path: str | Path, *, max_bytes: int = 1_048_576, max_items: int = 256):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.max_items = max_items
        if max_bytes <= 0 or max_items <= 0:
            raise ValueError("knowledge store limits must be positive")
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("knowledge path must not be a symlink")
        with self._lock:
            self._read()

    @staticmethod
    def _key(key: str) -> str:
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", key):
            raise ValueError("knowledge key must be 1-128 simple ASCII characters")
        return key

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        if self.path.stat().st_size > self.max_bytes:
            raise ValueError("knowledge store exceeds byte limit")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or len(data) > self.max_items:
            raise ValueError("invalid knowledge store")
        for key in data:
            self._key(key)
        return data

    def _write(self, data: dict[str, Any]) -> None:
        encoded = canonical_json(data)
        if len(data) > self.max_items or len(encoded.encode("utf-8")) > self.max_bytes:
            raise ValueError("knowledge store limit exceeded")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, prefix=".knowledge-", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(encoded)
                handle.flush()
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return copy.deepcopy(self._read().get(self._key(key), default))

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            data = self._read()
            data[self._key(key)] = copy.deepcopy(value)
            self._write(data)

    def delete(self, key: str) -> bool:
        with self._lock:
            data = self._read()
            key = self._key(key)
            existed = key in data
            data.pop(key, None)
            self._write(data)
            return existed

    def list(self, prefix: str = "") -> dict[str, Any]:
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a string")
        with self._lock:
            return {key: copy.deepcopy(value) for key, value in sorted(self._read().items()) if key.startswith(prefix)}
