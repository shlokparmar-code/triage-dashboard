"""Usage Mode helpers and configuration manager for Hospital vs. Home Use modes."""

import streamlit as st
from typing import Any, Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


def get_mode() -> str:
    """Return current usage mode ('🏥 Hospital' or '🏠 Home Use')."""
    if "usage_mode" not in st.session_state:
        st.session_state["usage_mode"] = config.DEFAULT_MODE
    return st.session_state["usage_mode"]


def is_hospital() -> bool:
    """Return True if currently in Hospital mode."""
    return get_mode() == config.MODE_HOSPITAL


def is_home() -> bool:
    """Return True if currently in Home Use mode."""
    return get_mode() == config.MODE_HOME


def get_mode_config() -> Dict[str, Any]:
    """Retrieve configuration dictionary for active mode."""
    mode = get_mode()
    return config.MODE_CONFIG.get(mode, config.MODE_CONFIG[config.DEFAULT_MODE])


def render_mode_selector() -> str:
    """Render top-of-page mode switcher without losing existing inputs or assessments."""
    current = get_mode()
    options = [config.MODE_HOME, config.MODE_HOSPITAL]
    idx = options.index(current) if current in options else 0

    col_mode, col_space = st.columns([1.5, 3.5])
    with col_mode:
        selected = st.radio(
            "Select Interface Environment:",
            options=options,
            index=idx,
            horizontal=True,
            key="_mode_radio_selector",
            help="Switch between Patient Home Self-Care and Hospital Clinical Triage Workstation."
        )

    if selected != current:
        st.session_state["usage_mode"] = selected
        st.rerun()

    return selected
