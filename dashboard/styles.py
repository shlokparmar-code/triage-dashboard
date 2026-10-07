"""CSS Styles, accessible high-contrast themes, and UI badges for the dashboard."""

CUSTOM_CSS = """
<style>
    /* Clean modern medical typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top banner styling */
    .urgency-banner-high {
        background: linear-gradient(135deg, #d90429 0%, #ef233c 100%);
        color: white;
        padding: 22px 26px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(217, 4, 41, 0.25);
    }

    .urgency-banner-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        color: #1f2937;
        padding: 22px 26px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.25);
    }

    .urgency-banner-low {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        padding: 22px 26px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
    }

    /* Card styling */
    .module-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }

    /* Chips */
    .chip {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 700;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .chip-high { background-color: #fee2e2; color: #b91c1c; border: 1px solid #f87171; }
    .chip-medium { background-color: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .chip-low { background-color: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }

    /* Highlighted text spans */
    .token-escalate {
        background-color: #fecaca;
        color: #991b1b;
        font-weight: 600;
        padding: 2px 5px;
        border-radius: 4px;
        border-bottom: 2px solid #ef4444;
    }
    .token-calm {
        background-color: #d1fae5;
        color: #065f46;
        font-weight: 500;
        padding: 2px 5px;
        border-radius: 4px;
        border-bottom: 2px solid #10b981;
    }

    /* Yoga & recommendation cards */
    .asana-card {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .diet-card {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #f9fafb;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #4b5563;
        margin-top: 24px;
        line-height: 1.4;
    }
</style>
"""
