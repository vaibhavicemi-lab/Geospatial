from database.db import get_alert_by_id, initialize_db, insert_alert, update_review


def test_database_insert_read_update(tmp_path):
    db_path = tmp_path / "alerts.db"
    initialize_db(db_path)
    alert = {
        "alert_id": "GI-TEST",
        "change_percentage": 5,
        "changed_pixels": 50,
        "number_of_regions": 2,
        "alert_level": "Medium",
        "confidence_score": 60,
        "analyst_status": "Pending",
        "analyst_remarks": "",
        "timestamp": "2026-01-01 10:00:00",
        "plain_english_summary": "Likely change detected.",
        "detected_change_type": "construction activity",
        "explanation_confidence": "Medium",
    }
    insert_alert(alert, "before.png", "after.png", db_path)
    saved = get_alert_by_id("GI-TEST", db_path)
    assert saved["before_image"] == "before.png"
    update_review("GI-TEST", "Accepted", "Confirmed", db_path)
    updated = get_alert_by_id("GI-TEST", db_path)
    assert updated["analyst_status"] == "Accepted"
    assert updated["analyst_remarks"] == "Confirmed"
