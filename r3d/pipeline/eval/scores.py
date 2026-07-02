# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Score:
    annotation_id: str
    model: str
    strategy: str
    parsed_answer: str | None
    gt_answer: str
    percentage_error: float | None
    accuracy: float | None
    parse_failed: bool
    created_ns: int


class ScoreStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS scores (
                annotation_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                strategy TEXT NOT NULL,
                parsed_answer TEXT,
                gt_answer TEXT NOT NULL,
                percentage_error REAL,
                accuracy REAL,
                parse_failed INTEGER NOT NULL,
                created_ns INTEGER NOT NULL
            )"""
        )
        self._conn.commit()

    def write(self, score: Score) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO scores
            (annotation_id, model, strategy, parsed_answer, gt_answer,
             percentage_error, accuracy, parse_failed, created_ns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                score.annotation_id,
                score.model,
                score.strategy,
                score.parsed_answer,
                score.gt_answer,
                score.percentage_error,
                score.accuracy,
                int(score.parse_failed),
                score.created_ns,
            ),
        )
        self._conn.commit()

    def get_all(self) -> list[Score]:
        rows = self._conn.execute(
            "SELECT * FROM scores ORDER BY annotation_id"
        ).fetchall()
        return [
            Score(
                annotation_id=row["annotation_id"],
                model=row["model"],
                strategy=row["strategy"],
                parsed_answer=row["parsed_answer"],
                gt_answer=row["gt_answer"],
                percentage_error=row["percentage_error"],
                accuracy=row["accuracy"],
                parse_failed=bool(row["parse_failed"]),
                created_ns=row["created_ns"],
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()
