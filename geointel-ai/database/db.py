import sqlite3
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE NOT NULL,
    before_image TEXT,
    after_image TEXT,
    change_percentage REAL,
    changed_pixels INTEGER,
    number_of_regions INTEGER,
    alert_level TEXT,
    confidence_score REAL,
    analyst_status TEXT,
    analyst_remarks TEXT,
    timestamp TEXT,
    plain_english_summary TEXT,
    detected_change_type TEXT,
    explanation_confidence TEXT
);
"""


def connect(db_path: Path | str = DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def initialize_db(db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(alerts)").fetchall()}
        migrations = {
            "plain_english_summary": "ALTER TABLE alerts ADD COLUMN plain_english_summary TEXT",
            "detected_change_type": "ALTER TABLE alerts ADD COLUMN detected_change_type TEXT",
            "explanation_confidence": "ALTER TABLE alerts ADD COLUMN explanation_confidence TEXT",
        }
        for column, sql in migrations.items():
            if column not in existing:
                conn.execute(sql)


def insert_alert(alert: Dict, before_image: str = "", after_image: str = "", db_path: Path | str = DB_PATH) -> str:
    initialize_db(db_path)
    fields = {
        "alert_id": alert["alert_id"],
        "before_image": before_image,
        "after_image": after_image,
        "change_percentage": alert.get("change_percentage", 0),
        "changed_pixels": alert.get("changed_pixels", 0),
        "number_of_regions": alert.get("number_of_regions", 0),
        "alert_level": alert.get("alert_level", "Low"),
        "confidence_score": alert.get("confidence_score", 0),
        "analyst_status": alert.get("analyst_status", "Pending"),
        "analyst_remarks": alert.get("analyst_remarks", ""),
        "timestamp": alert.get("timestamp", ""),
        "plain_english_summary": alert.get("plain_english_summary", ""),
        "detected_change_type": alert.get("detected_change_type", ""),
        "explanation_confidence": alert.get("explanation_confidence", ""),
    }
    columns = ", ".join(fields.keys())
    placeholders = ", ".join(["?"] * len(fields))
    with connect(db_path) as conn:
        conn.execute(f"INSERT OR REPLACE INTO alerts ({columns}) VALUES ({placeholders})", tuple(fields.values()))
    return fields["alert_id"]


def update_review(alert_id: str, status: str, remarks: str, db_path: Path | str = DB_PATH) -> None:
    initialize_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE alerts SET analyst_status = ?, analyst_remarks = ? WHERE alert_id = ?",
            (status, remarks, alert_id),
        )


def get_all_alerts(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    initialize_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM alerts ORDER BY id DESC", conn)


def get_alert_by_id(alert_id: str, db_path: Path | str = DB_PATH) -> Optional[Dict]:
    initialize_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        if row is None:
            return None
        columns = [col[1] for col in conn.execute("PRAGMA table_info(alerts)").fetchall()]
        return dict(zip(columns, row))
