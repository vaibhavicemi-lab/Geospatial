from datetime import datetime
from uuid import uuid4


def get_alert_level(change_percentage: float) -> str:
    if change_percentage < 2:
        return "Low"
    if change_percentage <= 10:
        return "Medium"
    if change_percentage <= 25:
        return "High"
    return "Critical"


def recommended_action(alert_level: str) -> str:
    actions = {
        "Low": "Log the change and review during routine monitoring.",
        "Medium": "Inspect the highlighted regions and compare with recent field knowledge.",
        "High": "Prioritize analyst review and validate whether major construction or land-use change occurred.",
        "Critical": "Escalate immediately for detailed geospatial intelligence review.",
    }
    return actions.get(alert_level, "Analyst review recommended.")


def generate_alert(analysis: dict, explanation: dict | None = None) -> dict:
    level = get_alert_level(float(analysis.get("change_percentage", 0)))
    return {
        "alert_id": "GI-" + uuid4().hex[:8].upper(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "change_percentage": analysis.get("change_percentage", 0),
        "changed_pixels": analysis.get("changed_pixels", 0),
        "number_of_regions": analysis.get("number_of_regions", 0),
        "alert_level": level,
        "confidence_score": analysis.get("confidence_score", 0),
        "recommended_action": recommended_action(level),
        "analyst_status": "Pending",
        "analyst_remarks": "",
        "plain_english_summary": (explanation or {}).get("main_summary", ""),
        "detected_change_type": (explanation or {}).get("detected_change_type", "unknown change"),
        "explanation_confidence": (explanation or {}).get("overall_confidence", "Low"),
    }
