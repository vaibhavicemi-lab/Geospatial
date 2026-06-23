import pandas as pd

from dashboard.report_generator import generate_csv_report, generate_pdf_report


def test_report_generation_outputs_bytes():
    rows = [{"alert_id": "GI-1", "alert_level": "Low", "change_percentage": 1.0, "number_of_regions": 1, "confidence_score": 40, "analyst_status": "Pending", "analyst_remarks": "", "timestamp": "now", "plain_english_summary": "No major change.", "detected_change_type": "unknown", "explanation_confidence": "Low"}]
    csv_bytes = generate_csv_report(pd.DataFrame(rows))
    pdf_bytes = generate_pdf_report(rows)
    assert b"GI-1" in csv_bytes
    assert pdf_bytes.startswith(b"%PDF")
