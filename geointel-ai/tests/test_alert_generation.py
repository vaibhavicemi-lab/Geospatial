from inference.alert_generation import generate_alert, get_alert_level


def test_alert_level_thresholds():
    assert get_alert_level(1.9) == "Low"
    assert get_alert_level(2.0) == "Medium"
    assert get_alert_level(10.0) == "Medium"
    assert get_alert_level(10.1) == "High"
    assert get_alert_level(25.1) == "Critical"


def test_generate_alert_includes_explanation_fields():
    alert = generate_alert({"change_percentage": 3, "changed_pixels": 10, "number_of_regions": 1, "confidence_score": 50}, {"main_summary": "Likely road extension detected.", "detected_change_type": "road extension", "overall_confidence": "Medium"})
    assert alert["alert_level"] == "Medium"
    assert alert["plain_english_summary"]
    assert alert["explanation_confidence"] == "Medium"
