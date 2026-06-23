from io import BytesIO
from typing import Dict, Iterable, List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REPORT_COLUMNS = [
    "before_image",
    "after_image",
    "alert_id",
    "alert_level",
    "change_percentage",
    "number_of_regions",
    "confidence_score",
    "analyst_status",
    "analyst_remarks",
    "timestamp",
    "plain_english_summary",
    "detected_change_type",
    "explanation_confidence",
]


def generate_csv_report(alerts_df: pd.DataFrame) -> bytes:
    export = alerts_df.copy()
    for column in REPORT_COLUMNS:
        if column not in export.columns:
            export[column] = ""
    return export[REPORT_COLUMNS].to_csv(index=False).encode("utf-8")


def generate_pdf_report(alerts: Iterable[Dict]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story: List = [Paragraph("GeoIntel AI Alert Report", styles["Title"]), Spacer(1, 12)]
    for alert in alerts:
        story.append(Paragraph(f"Alert ID: {alert.get('alert_id', '')}", styles["Heading2"]))
        rows = [
            ["Before Image", alert.get("before_image", "")],
            ["After Image", alert.get("after_image", "")],
            ["Alert Level", alert.get("alert_level", "")],
            ["Change %", str(alert.get("change_percentage", ""))],
            ["Regions", str(alert.get("number_of_regions", ""))],
            ["Confidence Score", str(alert.get("confidence_score", ""))],
            ["AI Explanation", alert.get("plain_english_summary", "")],
            ["Change Type", alert.get("detected_change_type", "")],
            ["Explanation Confidence", alert.get("explanation_confidence", "")],
            ["Analyst Status", alert.get("analyst_status", "")],
            ["Remarks", alert.get("analyst_remarks", "")],
            ["Timestamp", alert.get("timestamp", "")],
        ]
        table = Table(rows, colWidths=[130, 330])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.extend([table, Spacer(1, 16)])
    doc.build(story)
    return buffer.getvalue()
