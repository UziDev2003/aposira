"""Shared APOSIRA look and feel for every page (commerce, forecasting, vision)."""
import html
import streamlit as st

THEME_CSS = """<style>
.block-container {max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem;}
[data-testid="stAppViewContainer"] {background: #f8fafc;}
[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e2e8f0;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,.04);}
[data-testid="stForm"] {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.04);}
h1, h2, h3 {letter-spacing: -.025em; color: #0f172a;}
.aposira-hero {background: linear-gradient(125deg,#0f172a 0%,#1e3a8a 50%,#2563eb 100%); color: #fff;
    padding: 32px 36px; border-radius: 18px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(37,99,235,.15);}
.aposira-hero h1 {color: #fff; font-size: 2.1rem; font-weight: 700; margin: 8px 0 10px;}
.aposira-hero p {color: #dbeafe; font-size: 1.05rem; max-width: 720px; margin: 0; line-height: 1.5;}
.aposira-header {background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #2563eb;
    border-radius: 14px; padding: 18px 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.04);}
.aposira-header h1 {font-size: 1.7rem; font-weight: 700; margin: 4px 0 4px; padding: 0;}
.aposira-header p {color: #475569; margin: 0;}
.aposira-eyebrow {font-size: 11px; font-weight: 700; letter-spacing: .18em; color: #2563eb; text-transform: uppercase;}
.aposira-hero .aposira-eyebrow {color: #93c5fd;}
.status-badge {display: inline-block; padding: 3px 10px; font-size: 12px; font-weight: 600;
    border-radius: 9999px; text-transform: capitalize;}
.badge-completed {background:#dcfce7;color:#15803d;} .badge-dispatched {background:#dbeafe;color:#1d4ed8;}
.badge-ready {background:#f3e8ff;color:#7e22ce;} .badge-packed {background:#e0e7ff;color:#4338ca;}
.badge-picking {background:#fef3c7;color:#b45309;} .badge-accepted {background:#e0f2fe;color:#0369a1;}
.badge-received {background:#f1f5f9;color:#334155;} .badge-cancelled {background:#ffe4e6;color:#be123c;}
.badge-failed {background:#fee2e2;color:#b91c1c;}
.receipt-box {background:#fff; border:2px dashed #cbd5e1; border-radius:12px; padding:24px; margin-top:16px;}
/* Store Vision helpers, light variant */
.result-card {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; margin:10px 0;}
.zone {background:#eff6ff; color:#1e3a8a; padding:10px; border-radius:8px; margin:5px 0;}
.severity-LOW {color:#2563eb; font-weight:700;} .severity-MEDIUM {color:#d97706; font-weight:700;}
.severity-HIGH {color:#dc2626; font-weight:700;} .severity-CRITICAL {color:#7f1d1d; font-weight:700;}
</style>"""


def apply_theme():
    """Inject the shared APOSIRA stylesheet. Call once per script run."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def page_header(title, subtitle='', eyebrow='APOSIRA'):
    """Consistent page title card used by every workspace."""
    st.markdown(
        f'<div class="aposira-header"><div class="aposira-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )

