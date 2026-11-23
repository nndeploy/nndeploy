# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

from nndeploy.e_dit.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SECRET_KEYS = {
    "awssecretaccesskey",
    "password",
    "rootpass",
    "secretaccesskey",
    "secretkey",
}
_ACCESS_KEY_KEYS = {"accesskey", "awsaccesskeyid", "rootuser"}


def _normalized_sensitive_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _value_fingerprint(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _mask_access_key(value: Any) -> str:
    text = str(value)
    if len(text) <= 8:
        return "***"
    return f"{text[:4]}***{text[-4:]}"


def sanitize_videoedit_request_data(
    value: Any, *, include_sensitive_values: bool = False
) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            sensitive_key = _normalized_sensitive_key(key)
            if not include_sensitive_values and sensitive_key in _SECRET_KEYS:
                sanitized[key] = "***"
                if item is not None:
                    sanitized[f"{key}_sha256"] = _value_fingerprint(item)
                continue
            if not include_sensitive_values and sensitive_key in _ACCESS_KEY_KEYS:
                sanitized[key] = _mask_access_key(item) if item is not None else None
                if item is not None:
                    sanitized[f"{key}_sha256"] = _value_fingerprint(item)
                continue
            sanitized[key] = sanitize_videoedit_request_data(
                item, include_sensitive_values=include_sensitive_values
            )
        return sanitized

    if isinstance(value, (list, tuple)):
        return [
            sanitize_videoedit_request_data(
                item, include_sensitive_values=include_sensitive_values
            )
            for item in value
        ]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _safe_file_component(value: str | None) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "unknown")
    component = component.strip("._-")[:80]
    return component or "unknown"


class VideoEditRequestAudit:
    def __init__(
        self,
        log_dir: str | None,
        *,
        task_id: str | None = None,
        include_sensitive_values: bool = False,
    ):
        self.log_dir = os.path.abspath(os.path.expanduser(log_dir)) if log_dir else None
        self.include_sensitive_values = include_sensitive_values
        self.audit_id = uuid.uuid4().hex
        self.file_path: str | None = None
        self._record: dict[str, Any] = {}

        if self.log_dir is None:
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        task_component = _safe_file_component(task_id)
        file_name = f"{timestamp}_{task_component}_{self.audit_id[:8]}.request.json"
        self.file_path = os.path.join(self.log_dir, file_name)
        self._record = {
            "schema_version": 1,
            "audit_id": self.audit_id,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "sensitive_values_included": include_sensitive_values,
            "status": "received",
        }

    @property
    def enabled(self) -> bool:
        return self.file_path is not None

    def update(self, **fields: Any) -> str | None:
        if not self.enabled:
            return None

        self._record.update(
            sanitize_videoedit_request_data(
                fields,
                include_sensitive_values=self.include_sensitive_values,
            )
        )
        self._record["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self.file_path if self._write() else None

    def _write(self) -> bool:
        assert self.log_dir is not None
        assert self.file_path is not None
        temp_path = None
        try:
            os.makedirs(self.log_dir, mode=0o700, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(self.file_path)}.",
                suffix=".tmp",
                dir=self.log_dir,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                os.fchmod(file.fileno(), 0o600)
                json.dump(self._record, file, ensure_ascii=False, indent=2)
                file.write("\n")
            os.replace(temp_path, self.file_path)
            temp_path = None
            return True
        except (OSError, TypeError, ValueError) as error:
            logger.warning("Failed to write VideoEdit request audit: %s", error)
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
