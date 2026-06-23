from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import CLASSICAL_THRESHOLD, MIN_REGION_AREA, MODEL_WEIGHTS, SAMPLES_DIR, ensure_directories
from dashboard.report_generator import generate_csv_report, generate_pdf_report
from dashboard.ui_components import (
    inject_global_css,
    render_alert_card,
    render_empty_state,
    render_explanation_card,
    render_feature_card,
    render_footer,
    render_header,
    render_image_panel,
    render_metric_card,
    render_pipeline,
    status_badge_html,
)
from dashboard.visualizer import create_mask_preview, format_change_percentage
from database.db import get_all_alerts, initialize_db, insert_alert, update_review
from inference.alert_generation import generate_alert
from inference.change_detection import classical_change_detection, deep_learning_change_detection
from inference.change_explainer import explain_all_changes
from preprocessing.image_loader import load_uploaded_image


st.set_page_config(page_title="GeoIntel AI", page_icon="🛰️", layout="wide")
inject_global_css()


PAGES = {
    "Home": "home",
    "Upload & Analyze": "upload",
    "Alert History": "history",
    "Model Training": "training",
    "Reports": "reports",
    "About": "about",
}


def section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtle-text">{subtitle}</div>', unsafe_allow_html=True)


def create_demo_images():
    before_path = SAMPLES_DIR / "demo_before.png"
    after_path = SAMPLES_DIR / "demo_after.png"
    if before_path.exists() and after_path.exists():
        return before_path, after_path
    base = np.full((320, 420, 3), [86, 128, 82], dtype=np.uint8)
    cv2.rectangle(base, (40, 40), (380, 280), (110, 148, 96), -1)
    cv2.line(base, (20, 250), (400, 250), (115, 115, 110), 16)
    before = base.copy()
    after = base.copy()
    cv2.rectangle(after, (175, 95), (260, 170), (185, 185, 176), -1)
    cv2.rectangle(after, (275, 70), (360, 125), (58, 140, 83), -1)
    cv2.line(after, (260, 165), (400, 215), (130, 130, 125), 12)
    Image.fromarray(before).save(before_path)
    Image.fromarray(after).save(after_path)
    return before_path, after_path


def load_demo_pair():
    before_path, after_path = create_demo_images()
    before = np.array(Image.open(before_path).convert("RGB"))
    after = np.array(Image.open(after_path).convert("RGB"))
    return before, after, before_path.name, after_path.name


def run_analysis(before, after, mode, threshold, min_area):
    if mode == "Deep Learning Mode":
        result = deep_learning_change_detection(
            before,
            after,
            weights_path=str(MODEL_WEIGHTS),
            fallback_threshold=threshold,
            fallback_min_area=min_area,
        )
    else:
        result = classical_change_detection(before, after, threshold=threshold, min_area=min_area)
    explanation = explain_all_changes(result["before"], result["after"], result["mask"], result["analysis"]["regions"])
    alert = generate_alert(result["analysis"], explanation)
    result["explanation"] = explanation
    result["alert"] = alert
    return result


def get_dashboard_stats(alerts: pd.DataFrame) -> dict:
    if alerts.empty:
        return {"total": 0, "high_critical": 0, "avg_change": 0.0, "last_status": "None", "pending": 0, "accepted": 0}
    return {
        "total": len(alerts),
        "high_critical": int(alerts["alert_level"].isin(["High", "Critical"]).sum()),
        "avg_change": float(alerts["change_percentage"].fillna(0).mean()),
        "last_status": str(alerts.iloc[0].get("analyst_status", "Pending")),
        "pending": int((alerts["analyst_status"] == "Pending").sum()),
        "accepted": int((alerts["analyst_status"] == "Accepted").sum()),
    }


def render_top_nav() -> str:
    alerts = get_all_alerts()
    model_exists = MODEL_WEIGHTS.exists()
    model_badge = status_badge_html("Deep Learning Model Available", "success") if model_exists else status_badge_html("Classical Demo Mode Ready", "warning")
    page_names = list(PAGES.keys())
    if st.session_state.get("active_page") not in page_names:
        st.session_state["active_page"] = "Home"
    page = st.radio(
        "Main navigation",
        page_names,
        horizontal=True,
        label_visibility="collapsed",
        key="active_page",
    )
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-title">GeoIntel AI</div>
            <div class="topbar-subtitle">Satellite Change Detection • Alert Generation • Analyst Review</div>
            <div style="margin-top:0.55rem;">
                {status_badge_html("Local Processing", "success")}
                {status_badge_html("No Paid API", "info")}
                {status_badge_html("Analyst-in-the-Loop", "info")}
                {model_badge}
                {status_badge_html("SQLite Connected", "success")}
                {status_badge_html(f"Saved Alerts: {len(alerts)}", "muted")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return PAGES[page]


def navigate_to(page_name: str) -> None:
    st.session_state["active_page"] = page_name
    st.rerun()


def home_page():
    alerts = get_all_alerts()
    stats = get_dashboard_stats(alerts)

    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size:2.1rem; line-height:1; margin-bottom:1rem;">▲</div>
            <div class="main-header">GeoIntel AI</div>
            <h3 style="color:#fafafa; margin:0 auto 1rem auto; max-width:760px; font-size:1.45rem; font-weight:500;">
                AI-Based Geospatial Intelligence System for Satellite Image Change Detection
            </h3>
            <p class="subtle-text" style="max-width:680px; margin:0 auto 1.3rem auto;">
                Upload before-and-after satellite imagery, detect changed regions, generate alerts,
                and review results in a local analyst dashboard.
            </p>
            <div>
                <span class="status-badge" style="background:#ffffff;">🛰️ Satellite Imagery</span>
                <span class="status-badge" style="background:#e5e5e5;">🧠 Deep Learning Ready</span>
                <span class="status-badge" style="background:#d4d4d8;">🚨 Alert Generation</span>
                <span class="status-badge" style="background:#f4f4f5;">👨‍💻 Analyst Review</span>
                <span class="status-badge" style="background:#ffffff;">🔒 Local & Free</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Total Alerts", stats["total"], "Saved in SQLite", "🚨")
    with metric_cols[1]:
        render_metric_card("High/Critical", stats["high_critical"], "Priority investigations", "⚠️")
    with metric_cols[2]:
        render_metric_card("Average Change", format_change_percentage(stats["avg_change"]), "Mean alert area", "📊")
    with metric_cols[3]:
        render_metric_card("Last Alert Status", stats["last_status"], "Most recent review", "🧾")

    section_title("Operational Pipeline")
    render_pipeline()

    section_title("Core Capabilities")
    feature_cols = st.columns(3)
    features = [
        ("Image Upload", "Upload before and after satellite images with safe image handling.", "Upload & Analyze", "Open Upload"),
        ("Classical Change Detection", "Immediate OpenCV-based difference, thresholding, morphology, and contours.", "Upload & Analyze", "Run Analysis"),
        ("Siamese U-Net Ready", "Fine-tune and load trained PyTorch weights for deep learning masks.", "Model Training", "View Training"),
        ("Plain-English Explanation", "Rule-based summaries help examiners and analysts understand detected changes.", "Upload & Analyze", "Analyze Images"),
        ("Alert Review", "Save analyst status and remarks in SQLite for traceable review.", "Alert History", "Review Alerts"),
        ("CSV/PDF Reports", "Export alert history and selected records for final-year documentation.", "Reports", "Open Reports"),
    ]
    for index, feature in enumerate(features):
        with feature_cols[index % 3]:
            title, description, target_page, button_label = feature
            render_feature_card(title, description)
            if st.button(button_label, key=f"feature_action_{index}", width="stretch"):
                navigate_to(target_page)
    render_footer()


def file_info_card(title: str, name: str, image: np.ndarray, uploaded_file=None) -> None:
    size_text = f"{image.shape[1]} x {image.shape[0]} px"
    fmt = Path(name).suffix.upper().replace(".", "") or "PNG"
    file_size = "Demo image"
    if uploaded_file is not None:
        file_size = f"{uploaded_file.size / 1024:.1f} KB"
    st.markdown(
        f"""
        <div class="detail-card">
            <h4>{title}</h4>
            <p><b>File:</b> {name}</p>
            <p><b>Size:</b> {size_text} • <b>Format:</b> {fmt} • <b>File size:</b> {file_size}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def upload_analyze_page():
    render_header("Upload & Analyze", "Compare satellite images, detect changed regions, generate alerts, and save analyst review.")

    section_title("1. Upload Panel", "For best results, upload images of the same location with similar zoom, angle, and resolution.")
    use_demo = st.toggle("Use built-in demo satellite pair", value=True)
    before = after = None
    before_name = "demo_before.png"
    after_name = "demo_after.png"
    before_file = after_file = None

    upload_cols = st.columns(2)
    if use_demo:
        before, after, before_name, after_name = load_demo_pair()
        with upload_cols[0]:
            file_info_card("Before Image", before_name, before)
        with upload_cols[1]:
            file_info_card("After Image", after_name, after)
    else:
        with upload_cols[0]:
            with st.container(border=True):
                st.markdown("#### Before Image")
                before_file = st.file_uploader("Upload before image", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
        with upload_cols[1]:
            with st.container(border=True):
                st.markdown("#### After Image")
                after_file = st.file_uploader("Upload after image", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
        if before_file is None or after_file is None:
            render_empty_state("Waiting for image pair", "Upload both before and after images to begin analysis.", "🛰️")
            return
        try:
            before = load_uploaded_image(before_file)
            after = load_uploaded_image(after_file)
            before_name, after_name = before_file.name, after_file.name
            info_cols = st.columns(2)
            with info_cols[0]:
                file_info_card("Before Image", before_name, before, before_file)
            with info_cols[1]:
                file_info_card("After Image", after_name, after, after_file)
        except ValueError as exc:
            st.error(str(exc))
            return

    section_title("2. Processing Settings")
    settings_cols = st.columns([1.2, 1, 1, 0.8])
    with settings_cols[0]:
        mode = st.selectbox("Processing mode", ["Classical Demo Mode", "Deep Learning Mode"])
    with settings_cols[1]:
        threshold = st.slider("Classical threshold", 5, 100, CLASSICAL_THRESHOLD)
        st.caption("Lower values detect more changes but may increase false alerts.")
    with settings_cols[2]:
        min_area = st.slider("Minimum changed region area", 10, 1000, MIN_REGION_AREA)
        st.caption("Removes tiny noisy regions from the final mask.")
    with settings_cols[3]:
        st.write("")
        st.write("")
        run_clicked = st.button("Run Analysis", type="primary", width="stretch")

    section_title("3. Image Comparison")
    image_cols = st.columns(2)
    with image_cols[0]:
        render_image_panel("Before Image", before, before_name)
    with image_cols[1]:
        render_image_panel("After Image", after, after_name)

    if run_clicked:
        with st.spinner("Running geospatial change analysis..."):
            try:
                result = run_analysis(before, after, mode, threshold, min_area)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return
        st.session_state["last_result"] = result
        st.session_state["last_names"] = (before_name, after_name)

    result = st.session_state.get("last_result")
    if not result:
        render_empty_state("Analysis not started", "Choose settings and click Run Analysis to generate the mask, overlay, alert, and explanation.", "▶️")
        return

    if mode == "Deep Learning Mode" and not result.get("used_deep_learning"):
        st.warning(result["mode_message"])
    else:
        st.success(result["mode_message"])
    result_cols = st.columns(2)
    with result_cols[0]:
        render_image_panel("Change Mask", create_mask_preview(result["mask"]), "Red pixels indicate detected change.")
    with result_cols[1]:
        render_image_panel("Overlay Image", result["overlay"], "Changed regions highlighted on after image.")

    section_title("4. Analysis Results")
    analysis = result["analysis"]
    alert = result["alert"]
    metric_cols = st.columns(5)
    values = [
        ("Change %", f"{analysis['change_percentage']}%", "Changed image area", "📊"),
        ("Changed Pixels", analysis["changed_pixels"], "Binary mask count", "◼️"),
        ("Changed Regions", analysis["number_of_regions"], "Connected components", "📍"),
        ("Alert Level", alert["alert_level"], "Rule-based severity", "🚨"),
        ("Confidence", f"{analysis['confidence_score']}%", "Risk score", "🎯"),
    ]
    for col, args in zip(metric_cols, values):
        with col:
            render_metric_card(*args)

    section_title("5. AI Explanation")
    explanation = result["explanation"]
    render_explanation_card(explanation)
    if explanation.get("region_explanations"):
        with st.expander("Region-wise explanation details", expanded=True):
            st.dataframe(pd.DataFrame(explanation["region_explanations"]), width="stretch")

    section_title("6. Alert Card")
    render_alert_card(alert)

    section_title("7. Analyst Review")
    with st.container(border=True):
        statuses = ["Pending", "Accepted", "Rejected", "Needs Further Review"]
        status = st.selectbox("Analyst status", statuses)
        remarks = st.text_area("Analyst remarks", placeholder="Add interpretation, field notes, or reasons for accepting/rejecting the alert.")
        save_clicked = st.button("Save Review", type="primary", key="save_upload_review")
    if save_clicked:
        before_name, after_name = st.session_state.get("last_names", ("", ""))
        alert["analyst_status"] = status
        alert["analyst_remarks"] = remarks
        insert_alert(alert, before_name, after_name)
        st.success(f"Saved review for {alert['alert_id']}.")

    section_title("8. Export")
    export_alert = alert.copy()
    export_alert["before_image"] = st.session_state.get("last_names", ("", ""))[0]
    export_alert["after_image"] = st.session_state.get("last_names", ("", ""))[1]
    export_df = pd.DataFrame([export_alert])
    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button("Download Current Alert CSV", generate_csv_report(export_df), f"{alert['alert_id']}.csv", "text/csv", key="download_current_csv")
    with export_cols[1]:
        st.download_button("Download Current Alert PDF", generate_pdf_report([export_alert]), f"{alert['alert_id']}.pdf", "application/pdf", key="download_current_pdf")
    render_footer()


def color_history_table(df: pd.DataFrame):
    def level_color(value):
        colors = {"Low": "color:#22c55e;", "Medium": "color:#f59e0b;", "High": "color:#ef4444;", "Critical": "color:#fecaca; background-color:#7f1d1d;"}
        return colors.get(value, "")

    def status_color(value):
        colors = {"Pending": "color:#f59e0b;", "Accepted": "color:#22c55e;", "Rejected": "color:#ef4444;", "Needs Further Review": "color:#38bdf8;"}
        return colors.get(value, "")

    return df.style.map(level_color, subset=["Alert Level"]).map(status_color, subset=["Analyst Status"])


def filtered_alerts(alerts: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty:
        return alerts
    filters = st.columns([1, 1, 1, 1])
    with filters[0]:
        level = st.selectbox("Alert level", ["All"] + sorted(alerts["alert_level"].dropna().unique().tolist()))
    with filters[1]:
        status = st.selectbox("Analyst status", ["All"] + sorted(alerts["analyst_status"].dropna().unique().tolist()))
    with filters[2]:
        search = st.text_input("Search alert ID")
    with filters[3]:
        use_date = st.checkbox("Filter by date")
        selected_date = st.date_input("Date") if use_date else None

    filtered = alerts.copy()
    if level != "All":
        filtered = filtered[filtered["alert_level"] == level]
    if status != "All":
        filtered = filtered[filtered["analyst_status"] == status]
    if search:
        filtered = filtered[filtered["alert_id"].str.contains(search, case=False, na=False)]
    if selected_date:
        dates = pd.to_datetime(filtered["timestamp"], errors="coerce").dt.date
        filtered = filtered[dates == selected_date]
    return filtered


def alert_history_page():
    render_header("Alert History", "Review saved alerts, filter investigations, and update analyst decisions.")
    alerts = get_all_alerts()
    stats = get_dashboard_stats(alerts)
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Total Alerts", stats["total"], "All records", "🚨")
    with metric_cols[1]:
        render_metric_card("Pending", stats["pending"], "Awaiting review", "⏳")
    with metric_cols[2]:
        render_metric_card("Accepted", stats["accepted"], "Confirmed alerts", "✅")
    with metric_cols[3]:
        render_metric_card("High/Critical", stats["high_critical"], "Priority records", "⚠️")

    if alerts.empty:
        render_empty_state("No alerts generated yet", "Go to Upload & Analyze to run your first analysis.", "🚨")
        return

    section_title("Filters")
    filtered = filtered_alerts(alerts)
    if filtered.empty:
        render_empty_state("No matching alerts", "Adjust filters or clear the alert ID search.", "🔎")
        return

    display_cols = {
        "alert_id": "Alert ID",
        "alert_level": "Alert Level",
        "change_percentage": "Change %",
        "detected_change_type": "Detected Change Type",
        "plain_english_summary": "Plain-English Summary",
        "explanation_confidence": "Confidence",
        "analyst_status": "Analyst Status",
        "timestamp": "Timestamp",
    }
    display = filtered[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(color_history_table(display), width="stretch", hide_index=True)

    section_title("Selected Alert Details")
    selected = st.selectbox("Select alert", filtered["alert_id"].tolist())
    row = filtered[filtered["alert_id"] == selected].iloc[0].to_dict()
    render_alert_card(row)
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown(
            f"""
            <div class="detail-card">
                <h4>Metadata</h4>
                <p><b>Before:</b> {row.get('before_image', '')}</p>
                <p><b>After:</b> {row.get('after_image', '')}</p>
                <p><b>Changed pixels:</b> {row.get('changed_pixels', 0)}</p>
                <p><b>Timestamp:</b> {row.get('timestamp', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with detail_cols[1]:
        st.markdown(
            f"""
            <div class="detail-card">
                <h4>Review</h4>
                <p><b>Status:</b> {row.get('analyst_status', 'Pending')}</p>
                <p><b>Remarks:</b> {row.get('analyst_remarks', '') or 'No remarks added.'}</p>
                <p><b>Recommendation:</b> Analyst review recommended for final confirmation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    statuses = ["Pending", "Accepted", "Rejected", "Needs Further Review"]
    current_status = row["analyst_status"] if row.get("analyst_status") in statuses else "Pending"
    update_cols = st.columns([1, 2, 0.8])
    with update_cols[0]:
        status = st.selectbox("Update status", statuses, index=statuses.index(current_status))
    with update_cols[1]:
        remarks = st.text_area("Update remarks", value=row.get("analyst_remarks", "") or "")
    with update_cols[2]:
        st.write("")
        st.write("")
        if st.button("Update Review", type="primary", key="history_update_review"):
            update_review(selected, status, remarks)
            st.success("Review updated.")

    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button("Export Filtered CSV", generate_csv_report(filtered), "geointel_filtered_alerts.csv", "text/csv", key="history_filtered_csv")
    with export_cols[1]:
        st.download_button("Export Selected PDF", generate_pdf_report([row]), f"{selected}.pdf", "application/pdf", key="history_selected_pdf")
    render_footer()


def model_training_page():
    render_header("Model Training", "Fine-tune Siamese U-Net on LEVIR-CD style before/after satellite imagery.")
    section_title("A. Model Overview")
    tabs = st.tabs(["Classical Mode", "Siamese U-Net", "U-Net Baseline", "Future Transformer"])
    tabs[0].write("Classical mode uses image difference, thresholding, morphology, and contours. It works immediately without weights.")
    tabs[1].write("Siamese U-Net uses a shared encoder for before/after images, feature differences, and a decoder that outputs a sigmoid change mask.")
    tabs[2].write("The U-Net baseline concatenates before and after images as a 6-channel input for optional comparison.")
    tabs[3].write("SegFormer or Vision Transformer models can be added later for stronger semantic understanding.")

    section_title("B. Dataset Format")
    st.code(
        """LEVIR-CD/
  train/
    A/
    B/
    label/
  val/
    A/
    B/
    label/
  test/
    A/
    B/
    label/""",
        language="text",
    )

    section_title("C. Training Command")
    st.code(
        """python training/train_siamese_unet.py \\
  --data_dir data/LEVIR-CD \\
  --epochs 30 \\
  --batch_size 4 \\
  --image_size 256 \\
  --lr 0.0001 \\
  --save_path models/weights/siamese_unet_best.pth""",
        language="bash",
    )

    section_title("D. Metrics Explanation")
    metric_cols = st.columns(3)
    metrics = [
        ("IoU", "Intersection over Union between predicted and true changed pixels.", "∩"),
        ("Precision", "How many predicted changed pixels are actually changed.", "🎯"),
        ("Recall", "How many true changed pixels the model found.", "🔎"),
        ("F1-score", "Balanced score between precision and recall.", "⚖️"),
        ("Dice score", "Segmentation overlap score, useful for masks.", "◐"),
        ("Accuracy", "Overall pixel-level correctness.", "✅"),
    ]
    for index, metric in enumerate(metrics):
        with metric_cols[index % 3]:
            render_feature_card(*metric)

    section_title("E. Model Status")
    if MODEL_WEIGHTS.exists():
        st.markdown(status_badge_html("Deep Learning Model Available", "success"), unsafe_allow_html=True)
        st.caption(str(MODEL_WEIGHTS))
    else:
        st.markdown(status_badge_html("Model weights not found. Classical Demo Mode is available.", "warning"), unsafe_allow_html=True)
        st.caption(str(MODEL_WEIGHTS))
    render_footer()


def reports_page():
    render_header("Reports", "Export alert history and selected alert records for documentation.")
    alerts = get_all_alerts()
    if alerts.empty:
        render_empty_state("No report data yet", "Run an analysis and save an analyst review to generate report exports.", "📄")
        return

    stats = get_dashboard_stats(alerts)
    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Report Records", stats["total"], "Saved alerts available", "📄")
    with cols[1]:
        render_metric_card("Priority Alerts", stats["high_critical"], "High or critical", "⚠️")
    with cols[2]:
        render_metric_card("Average Change", format_change_percentage(stats["avg_change"]), "Across saved alerts", "📊")

    section_title("Export Options")
    selected = st.selectbox("Select alert for PDF", alerts["alert_id"].tolist())
    row = alerts[alerts["alert_id"] == selected].iloc[0].to_dict()
    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button("Export All Alerts as CSV", generate_csv_report(alerts), "geointel_all_alerts.csv", "text/csv", key="reports_all_csv")
    with export_cols[1]:
        st.download_button("Export Selected Alert as PDF", generate_pdf_report([row]), f"{selected}.pdf", "application/pdf", key="reports_selected_pdf")

    section_title("Sample Report Structure")
    st.markdown(
        """
        <div class="detail-card">
            <p><b>Project name:</b> GeoIntel AI</p>
            <p><b>Alert metadata:</b> alert ID, timestamp, image names, alert level</p>
            <p><b>Change analysis:</b> changed pixels, change percentage, region count, confidence score</p>
            <p><b>AI explanation:</b> plain-English summary, detected change type, explanation confidence</p>
            <p><b>Analyst review:</b> review status and remarks</p>
            <p><b>Limitations note:</b> AI output is decision-support and requires analyst confirmation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer()


def about_page():
    render_header("About GeoIntel AI", "A final-year project dashboard for local geospatial intelligence workflows.")
    sections = {
        "Abstract": "GeoIntel AI compares before-and-after satellite images to detect changed regions, generate alerts, support analyst review, and export reports.",
        "Problem Statement": "Manual satellite image comparison is slow, inconsistent, and difficult to document. This system provides a structured decision-support workflow.",
        "Objectives": "Detect visual changes, generate risk-based alerts, explain changes in simple English, save analyst review, and export CSV/PDF reports.",
        "Technology Stack": "Python, Streamlit, OpenCV, PyTorch, SQLite, Pandas, Pillow, Matplotlib, and ReportLab.",
        "System Architecture": "Satellite Images → Preprocessing → Change Detection → AI Explanation → Alert Generation → Analyst Review → Report Export.",
        "Applications": "Urban growth monitoring, disaster assessment, infrastructure monitoring, environmental surveillance, and defence-style geospatial intelligence demonstration.",
        "Limitations": "Accuracy depends on image alignment. Plain-English explanation is not final proof. Low-resolution images reduce classification quality. Human analyst review is required.",
        "Future Scope": "GeoTIFF support, map coordinate output, real-world area calculation, trained land-use classification, transformer models, and multi-user analyst workflows.",
    }
    for title, text in sections.items():
        st.markdown(f'<div class="detail-card"><h4>{title}</h4><p class="subtle-text">{text}</p></div>', unsafe_allow_html=True)
    render_footer()


def main():
    ensure_directories()
    initialize_db()
    create_demo_images()
    page = render_top_nav()
    if page == "home":
        home_page()
    elif page == "upload":
        upload_analyze_page()
    elif page == "history":
        alert_history_page()
    elif page == "training":
        model_training_page()
    elif page == "reports":
        reports_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
