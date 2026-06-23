from html import escape
from typing import Any, Dict, Optional

import streamlit as st


ALERT_COLORS = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#ef4444",
    "Critical": "#7f1d1d",
}

STATUS_COLORS = {
    "success": "#ffffff",
    "warning": "#ffffff",
    "danger": "#ffffff",
    "info": "#ffffff",
    "muted": "#111111",
}


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #000000;
            --panel: #0a0a0a;
            --panel-2: #111111;
            --border: rgba(255, 255, 255, 0.14);
            --text: #fafafa;
            --muted: #a1a1aa;
            --cyan: #ededed;
            --blue: #ffffff;
            --green: #22c55e;
            --orange: #f59e0b;
            --red: #ef4444;
        }
        .stApp {
            background: var(--bg);
            color: var(--text);
        }
        header[data-testid="stHeader"] {
            display: none;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        #MainMenu {
            visibility: hidden;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #06101f 0%, #080d18 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] * {
            color: #e5f3ff;
        }
        .block-container {
            padding-top: 0;
            padding-bottom: 2rem;
            max-width: 1440px;
        }
        .nav-spacer {
            height: 0.2rem;
        }
        .topbar {
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 0;
            border-left: 0;
            border-right: 0;
            padding: 1rem 1.15rem;
            margin-bottom: 0.75rem;
            box-shadow: none;
        }
        .topbar-title {
            font-size: 1.45rem;
            font-weight: 900;
            color: #f8fafc;
            margin: 0;
        }
        .topbar-subtitle {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }
        div[role="radiogroup"] {
            gap: 0.35rem;
            flex-wrap: wrap;
            background: transparent;
            border: 0;
            border-bottom: 1px solid var(--border);
            border-radius: 0;
            padding: 0.7rem 0 0.65rem 0;
            margin-bottom: 0;
        }
        div[role="radiogroup"] label {
            background: #000000;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            padding: 0.52rem 1rem;
            margin-right: 0.22rem;
            min-height: 2.5rem;
            color: #ffffff;
            font-weight: 600;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }
        div[role="radiogroup"] label:hover {
            border-color: rgba(255, 255, 255, 0.55);
            background: #111111;
        }
        div[role="radiogroup"] label:has(input:checked) {
            background: #ffffff;
            color: #000000;
            border-color: #ffffff;
        }
        div[role="radiogroup"] label:has(input:checked) * {
            color: #000000 !important;
        }
        div[role="radiogroup"] label > div:first-child {
            display: none;
        }
        .main-header {
            font-size: clamp(3.25rem, 8vw, 6.6rem);
            font-weight: 850;
            color: var(--text);
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.05em;
            line-height: 0.96;
        }
        .hero-card {
            background: transparent;
            border: 0;
            border-bottom: 1px solid var(--border);
            border-radius: 0;
            padding: 5rem 1rem 4.25rem 1rem;
            box-shadow: none;
            margin-bottom: 1.25rem;
            text-align: center;
        }
        .metric-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
            min-height: 116px;
        }
        .metric-card .metric-title {
            color: var(--muted);
            font-size: 0.86rem;
            margin-bottom: 0.35rem;
        }
        .metric-card .metric-value {
            color: var(--text);
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.15;
        }
        .metric-card .metric-subtitle {
            color: var(--muted);
            font-size: 0.84rem;
            margin-top: 0.35rem;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.32rem 0.68rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            color: #000000;
            font-weight: 700;
            font-size: 0.78rem;
            background: #ffffff;
            border: 1px solid rgba(255,255,255,0.28);
        }
        .status-badge.status-badge-muted {
            color: #ffffff;
            background: #000000;
            border: 1px solid rgba(255,255,255,0.28);
        }
        .section-title {
            color: var(--text);
            font-size: 1.25rem;
            font-weight: 800;
            margin: 1.2rem 0 0.6rem 0;
        }
        .subtle-text {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .image-panel, .explanation-card, .review-card, .feature-card, .pipeline-card, .detail-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.22);
            margin-bottom: 0.85rem;
        }
        .image-panel h4, .explanation-card h4, .review-card h4, .feature-card h4 {
            margin: 0 0 0.5rem 0;
            color: var(--text);
            font-size: 1rem;
        }
        .alert-card-low, .alert-card-medium, .alert-card-high, .alert-card-critical {
            border-radius: 12px;
            padding: 1.1rem;
            margin: 0.9rem 0;
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.22);
        }
        .alert-card-low { border-left: 8px solid var(--green); }
        .alert-card-medium { border-left: 8px solid var(--orange); }
        .alert-card-high { border-left: 8px solid var(--red); }
        .alert-card-critical { border-left: 8px solid #7f1d1d; }
        .alert-title {
            color: var(--text);
            font-size: 1.22rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .pipeline-step {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.8rem;
            text-align: center;
            color: var(--text);
            min-height: 74px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
        }
        .footer {
            color: var(--muted);
            border-top: 1px solid var(--border);
            margin-top: 2rem;
            padding-top: 1rem;
            font-size: 0.86rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #ffffff;
            background: #ffffff;
            color: #000000;
            font-weight: 800;
            min-height: 2.75rem;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #d4d4d8;
            background: #e5e5e5;
            color: #000000;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: var(--panel);
            border-radius: 10px;
            border: 1px solid var(--border);
        }
        code, pre {
            border-radius: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: Optional[str] = None) -> None:
    st.markdown(f'<div class="main-header">{escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtle-text">{escape(subtitle)}</div>', unsafe_allow_html=True)


def render_metric_card(title: str, value: Any, subtitle: Optional[str] = None, icon: Optional[str] = None) -> None:
    icon_html = f"{escape(icon)} " if icon else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{icon_html}{escape(title)}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-subtitle">{escape(subtitle or "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(text: str, status: str = "info") -> None:
    class_name = "status-badge status-badge-muted" if status == "muted" else "status-badge"
    st.markdown(
        f'<span class="{class_name}">{escape(text)}</span>',
        unsafe_allow_html=True,
    )


def status_badge_html(text: str, status: str = "info") -> str:
    class_name = "status-badge status-badge-muted" if status == "muted" else "status-badge"
    return f'<span class="{class_name}">{escape(text)}</span>'


def render_alert_card(alert_data: Dict[str, Any]) -> None:
    level = str(alert_data.get("alert_level", "Low"))
    class_name = f"alert-card-{level.lower()}" if level in ALERT_COLORS else "alert-card-low"
    st.markdown(
        f"""
        <div class="{class_name}">
            <div class="alert-title">{escape(level)} Alert - {escape(str(alert_data.get("alert_id", "")))}</div>
            <div class="subtle-text">
                <b>Change:</b> {escape(str(alert_data.get("change_percentage", 0)))}% &nbsp; | &nbsp;
                <b>Regions:</b> {escape(str(alert_data.get("number_of_regions", 0)))} &nbsp; | &nbsp;
                <b>Confidence:</b> {escape(str(alert_data.get("confidence_score", 0)))}%
            </div>
            <p><b>AI Explanation:</b> {escape(str(alert_data.get("plain_english_summary", "No explanation available.")))}</p>
            <p><b>Recommended Action:</b> {escape(str(alert_data.get("recommended_action", "Analyst review recommended.")))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation_card(explanation_data: Dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="explanation-card">
            <h4>AI Explanation</h4>
            <p><b>Main summary:</b> {escape(str(explanation_data.get("main_summary", "No summary available.")))}</p>
            <p><b>Detected change type:</b> {escape(str(explanation_data.get("detected_change_type", "unknown change")))}</p>
            <p><b>Explanation confidence:</b> {escape(str(explanation_data.get("overall_confidence", "Low")))}</p>
            <p><b>Analyst recommendation:</b> {escape(str(explanation_data.get("analyst_recommendation", "Analyst review recommended.")))}</p>
            <p class="subtle-text">AI explanation is decision-support only. Final confirmation requires analyst review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_panel(title: str, image, caption: Optional[str] = None) -> None:
    st.markdown(
        f'<div class="image-panel"><h4>{escape(title)}</h4><div class="subtle-text">{escape(caption or "")}</div></div>',
        unsafe_allow_html=True,
    )
    st.image(image, caption=caption, width="stretch", clamp=True)


def render_empty_state(title: str, message: str, icon: str = "ℹ️") -> None:
    st.markdown(
        f"""
        <div class="detail-card" style="text-align:center; padding:2rem;">
            <div style="font-size:2rem;">{escape(icon)}</div>
            <h3>{escape(title)}</h3>
            <p class="subtle-text">{escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        '<div class="footer">GeoIntel AI • Local satellite image change detection • Built for final-year project demonstration</div>',
        unsafe_allow_html=True,
    )


def render_pipeline() -> None:
    steps = [
        "Satellite Images",
        "Preprocessing",
        "Change Detection",
        "AI Explanation",
        "Alert Generation",
        "Analyst Review",
        "Report Export",
    ]
    columns = st.columns(len(steps))
    for col, step in zip(columns, steps):
        with col:
            st.markdown(f'<div class="pipeline-step">{escape(step)}</div>', unsafe_allow_html=True)


def render_feature_card(title: str, description: str, icon: str = "") -> None:
    heading = f"{escape(icon)} {escape(title)}" if icon else escape(title)
    st.markdown(
        f"""
        <div class="feature-card">
            <h4>{heading}</h4>
            <p class="subtle-text">{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(total_alerts: int, high_critical: int, last_alert: str, avg_change: float) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Alerts", total_alerts, "Saved review records", "🚨")
    with c2:
        render_metric_card("High/Critical", high_critical, "Priority alerts", "⚠️")
    with c3:
        render_metric_card("Last Alert", last_alert or "None", "Most recent alert ID", "🕒")
    with c4:
        render_metric_card("Avg Change %", f"{avg_change:.2f}", "Across all alerts", "📊")


alert_card = render_alert_card
