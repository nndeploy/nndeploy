 # db.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import sqlite3
import uuid
import logging

@dataclass
class WorkflowMeta:
    id: str
    path: str
    name: Optional[str]
    ext: Optional[str]
    size: Optional[int]
    cover: Optional[str]
    requirements: Optional[str]

@dataclass
class TemplateMeta:
    id: str
    path: str
    name: Optional[str]
    ext: Optional[str]
    size: Optional[int]
    cover: Optional[str]
    requirements: Optional[str]
    category: Optional[str]

class DB:
    """
    Thin wrapper around sqlite for workflows/templates.
    Thread-safe for FastAPI (uses check_same_thread=False).
    Keep SQL in one place.
    """
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    # ---------- schema ----------
    def init_schema(self) -> "DB":
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            name TEXT,
            ext TEXT,
            size INTEGER,
            cover TEXT,
            requirements TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_workflows_path ON workflows(path)""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            name TEXT,
            ext TEXT,
            size INTEGER,
            cover TEXT,
            requirements TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_templates_path ON templates(path)""")
        self.conn.commit()
        return self

    # ---------- helpers ----------
    @staticmethod
    def _stat(file_path: Path) -> Tuple[str, str, int | None]:
        name = file_path.name
        ext = file_path.suffix.lower()
        size = file_path.stat().st_size if file_path.exists() else None
        return name, ext, size

    # ---------- workflows ----------
    def insert_workflow(self,
                        name: str,
                        file_path: Path,
                        cover: Optional[str] = None,
                        requirements: Optional[str] = None) -> str:
        """
        Insert workflow row for a *new* file. Returns id (uuid).
        """
        wid = str(uuid.uuid4())
        _, ext, size = self._stat(file_path)
        abs_path = str(file_path.resolve())

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO workflows (id, path, name, ext, size, cover, requirements)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (wid, abs_path, name, ext, size, cover, requirements)
        )
        self.conn.commit()
        return wid

    def update_workflow_metadata(self, id: str, name: str, file_path: Path,
                                 cover: Optional[str] = None,
                                 requirements: Optional[str] = None) -> None:
        """
        Update metadata (name/ext/size/cover/requirements) of a workflow row by id.
        """
        _, ext, size = self._stat(file_path)
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE workflows
            SET name = ?, ext = ?, size = ?, cover = ?, requirements = ?
            WHERE id = ?
            """,
            (name, ext, size, cover, requirements, id)
        )
        self.conn.commit()

    def upsert_workflow_by_path(self, file_path: Path, name: str,
                                cover: Optional[str] = None,
                                requirements: Optional[str] = None) -> str:
        """
        Ensure a row exists for the given file path; if exists, update metadata;
        if not, insert a new row. Returns id.
        """
        abs_path = str(file_path.resolve())

        cur = self.conn.cursor()
        row = cur.execute("SELECT id FROM workflows WHERE path = ?", (abs_path,)).fetchone()
        if row:
            wid = row["id"]
            try:
                self.update_workflow_metadata(wid, name, file_path, cover, requirements)
            except Exception as e:
                logging.warning(f"[DB.upsert_workflow_by_path] update failed for {abs_path}: {e}")
            return wid
        else:
            try:
                return self.insert_workflow(name, file_path, cover, requirements)
            except Exception as e:
                logging.warning(f"[DB.upsert_workflow_by_path] insert failed for {abs_path}: {e}")
                raise

    def get_workflow_path(self, id_: str) -> Optional[Path]:
        cur = self.conn.cursor()
        row = cur.execute("SELECT path FROM workflows WHERE id = ?", (id_,)).fetchone()
        return Path(row["path"]) if row else None

    def get_workflow_name(self, id_: str) -> Optional[Path]:
        cur = self.conn.cursor()
        row = cur.execute("SELECT name FROM workflows WHERE id = ?", (id_,)).fetchone()
        return Path(row["name"]) if row else None

    def get_workflow_id_and_cover_by_path(self, file_path: Path) -> Optional[Tuple[str, Optional[str]]]:
        abs_path = str(file_path.resolve())
        cur = self.conn.cursor()
        row = cur.execute("SELECT id, cover FROM workflows WHERE path = ?", (abs_path,)).fetchone()
        if row:
            return row["id"], row["cover"]
        return None

    def delete_workflow(self, id_: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM workflows WHERE id = ?", (id_,))
        self.conn.commit()
        return cur.rowcount > 0

    # ---------- templates ----------
    def insert_or_ignore_template(self, file_path: Path,
                                  cover: Optional[str] = None,
                                  category: Optional[str] = None,
                                  requirements: Optional[str] = None) -> None:
        tid = str(uuid.uuid4())
        name, ext, size = self._stat(file_path)
        abs_path = str(file_path.resolve())

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO templates (id, path, name, ext, size, cover, requirements, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (tid, abs_path, name, ext, size, cover, requirements, category)
        )
        self.conn.commit()

    def get_template_path(self, id_: str) -> Optional[Path]:
        cur = self.conn.cursor()
        row = cur.execute("SELECT path FROM templates WHERE id = ?", (id_,)).fetchone()
        return Path(row["path"]) if row else None

    def get_template_meta_by_path(self, file_path: Path) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        """
        Returns (id, cover, requirements, category) for a template path.
        """
        abs_path = str(file_path.resolve())
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT id, cover, requirements, category FROM templates WHERE path = ?",
            (abs_path,)
        ).fetchone()
        if row:
            return row["id"], row["cover"], row["requirements"], row["category"]
        return None

    # ---------- raw access (only if really needed) ----------
    def cursor(self) -> sqlite3.Cursor:
        return self.conn.cursor()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
