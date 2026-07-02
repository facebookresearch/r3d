# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Response:
    annotation_id: str
    model: str
    strategy: str
    response: str
    tool_call_log: list[dict] | None
    created_ns: int
    latency_s: float = 0.0


class ResponseStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                annotation_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                strategy TEXT NOT NULL,
                response TEXT NOT NULL,
                tool_call_log TEXT,
                created_ns INTEGER NOT NULL,
                latency_s REAL NOT NULL DEFAULT 0.0
            )"""
        )
        self._conn.commit()

    def get(self, annotation_id: str) -> Response | None:
        row = self._conn.execute(
            "SELECT * FROM responses WHERE annotation_id = ?",
            (annotation_id,),
        ).fetchone()
        if row is None:
            return None
        return Response(
            annotation_id=row["annotation_id"],
            model=row["model"],
            strategy=row["strategy"],
            response=row["response"],
            tool_call_log=(
                json.loads(row["tool_call_log"])
                if row["tool_call_log"] is not None
                else None
            ),
            created_ns=row["created_ns"],
            latency_s=row["latency_s"],
        )

    def write(self, response: Response) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO responses
            (annotation_id, model, strategy, response, tool_call_log, created_ns, latency_s)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                response.annotation_id,
                response.model,
                response.strategy,
                response.response,
                json.dumps(response.tool_call_log)
                if response.tool_call_log is not None
                else None,
                response.created_ns,
                response.latency_s,
            ),
        )
        self._conn.commit()

    def get_all(self) -> list[Response]:
        rows = self._conn.execute(
            "SELECT * FROM responses ORDER BY annotation_id"
        ).fetchall()
        return [
            Response(
                annotation_id=row["annotation_id"],
                model=row["model"],
                strategy=row["strategy"],
                response=row["response"],
                tool_call_log=(
                    json.loads(row["tool_call_log"])
                    if row["tool_call_log"] is not None
                    else None
                ),
                created_ns=row["created_ns"],
                latency_s=row["latency_s"],
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()
