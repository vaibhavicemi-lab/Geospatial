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
