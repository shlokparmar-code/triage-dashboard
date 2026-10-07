"""Reusable UI components, chart plotters, and explainability visualizers for Streamlit."""

import re
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render_urgency_chip(urgency: str) -> str:
    """Return HTML string for urgency badge."""
    u_lower = urgency.lower()
    return f'<span class="chip chip-{u_lower}">{urgency}</span>'


def render_urgency_banner(
    urgency: str,
    score: float,
    synthesis: str,
    is_override: bool = False,
    simple_language: bool = False
) -> str:
    """Generate high-contrast top-line urgency banner."""
    u_lower = urgency.lower()
    
    if simple_language:
        title_map = {
            "HIGH": "CRITICAL EMERGENCY - GO TO HOSPITAL IMMEDIATELY",
            "MEDIUM": "ATTENTION NEEDED - SEE A CLINIC DOCTOR SOON",
            "LOW": "SAFE - REST AND TAKE CARE AT HOME"
        }
        lead_title = title_map.get(urgency, urgency)
    else:
        lead_title = f"{urgency} URGENCY — TRIAGE PROTOCOL"

    override_tag = '<span style="background: rgba(0,0,0,0.25); padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; margin-left: 10px;">SAFETY OVERRIDE ACTIVE</span>' if is_override else ""

    banner_html = f"""
    <div class="urgency-banner-{u_lower}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h2 style="margin: 0; font-size: 1.55rem; font-weight: 700;">{lead_title} {override_tag}</h2>
            <div style="font-size: 1.8rem; font-weight: 800; opacity: 0.95;">Score: {score:.0%}</div>
        </div>
        <p style="margin: 0; font-size: 1.02rem; line-height: 1.5; opacity: 0.95;">
            {synthesis}
        </p>
    </div>
    """
    return banner_html


def highlight_symptom_tokens(raw_text: str, top_tokens: List[Dict[str, Any]]) -> str:
    """Inject color-coded highlight spans around influential clinical tokens in raw symptom text."""
    if not raw_text:
        return "<em>No symptoms provided.</em>"

    if not top_tokens:
        return f"<div style='line-height: 1.8; font-size: 1.05rem;'>{raw_text}</div>"

    annotated = raw_text
    # Sort tokens longest first to avoid partial sub-word replacement collisions
    sorted_tokens = sorted(top_tokens, key=lambda x: len(x.get("token", "")), reverse=True)

    for item in sorted_tokens:
        token = item.get("token", "")
        # Clean potential n-gram / underscore notation
        clean_token = token.replace("not_", "no ").replace("_", " ").strip()
        if not clean_token or len(clean_token) < 3:
            continue

        weight = item.get("weight", 0.0)
        css_class = "token-escalate" if weight > 0 else "token-calm"
        polarity = f"Impact: +{weight:.2f}" if weight > 0 else f"Impact: {weight:.2f}"

        # Case-insensitive replacement with boundary preservation
        pattern = re.compile(rf"\b({re.escape(clean_token)})\b", re.IGNORECASE)
        replacement = f'<span class="{css_class}" title="{polarity}">\\1</span>'
        annotated = pattern.sub(replacement, annotated)

    return f"<div style='line-height: 1.9; font-size: 1.05rem;'>{annotated}</div>"


def plot_vitals_telemetry(df: pd.DataFrame, flagged_indices: Optional[List[int]] = None):
    """Plot multi-variate telemetry charts (Heart Rate and SpO2) with anomaly intervals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)

    time_idx = range(len(df))

    # --- 1. Heart Rate Plot ---
    ax1.plot(time_idx, df["heart_rate"], color="#2563eb", linewidth=1.6, label="Heart Rate (bpm)")
    ax1.axhspan(60, 100, color="#10b981", alpha=0.12, label="Normal Range (60-100)")
    ax1.axhline(140, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.8, label="Tachycardia Critical (>140)")
    ax1.axhline(40, color="#9333ea", linestyle="--", linewidth=1.0, alpha=0.8, label="Bradycardia Critical (<40)")

    # Highlight anomalous regions
    if "ground_truth_anomaly" in df.columns and df["ground_truth_anomaly"].sum() > 0:
        anom_mask = df["ground_truth_anomaly"] == 1
        ax1.fill_between(time_idx, df["heart_rate"].min() - 5, df["heart_rate"].max() + 5,
                         where=anom_mask, color="#ef4444", alpha=0.22, label="Flagged Anomaly Window")
    elif flagged_indices:
        anom_mask = [i in flagged_indices for i in range(len(df))]
        ax1.fill_between(time_idx, df["heart_rate"].min() - 5, df["heart_rate"].max() + 5,
                         where=anom_mask, color="#ef4444", alpha=0.22, label="Flagged Anomaly Window")

    ax1.set_ylabel("Heart Rate (bpm)", fontsize=9, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax1.tick_params(labelsize=8)

    # --- 2. SpO2 Plot ---
    ax2.plot(time_idx, df["spo2"], color="#0891b2", linewidth=1.6, label="SpO2 (%)")
    ax2.axhspan(94, 100, color="#10b981", alpha=0.12, label="Normal Oxygen (94-100%)")
    ax2.axhline(90, color="#dc2626", linestyle="--", linewidth=1.1, label="Critical Hypoxia (<90%)")
    
    if "ground_truth_anomaly" in df.columns and df["ground_truth_anomaly"].sum() > 0:
        anom_mask = df["ground_truth_anomaly"] == 1
        ax2.fill_between(time_idx, 75, 101, where=anom_mask, color="#ef4444", alpha=0.22)
    elif flagged_indices:
        anom_mask = [i in flagged_indices for i in range(len(df))]
        ax2.fill_between(time_idx, 75, 101, where=anom_mask, color="#ef4444", alpha=0.22)

    ax2.set_ylabel("SpO2 (%)", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Time Steps (Sequential Readings)", fontsize=9)
    ax2.set_ylim(min(78, df["spo2"].min() - 2), 101)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    return fig
