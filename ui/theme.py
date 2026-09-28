"""
ui/theme.py
-----------
Centralised colour palette and font constants for the application.
Import Theme anywhere in the UI layer instead of scattering hex strings.
"""


class Theme:
    # backgrounds
    DARK_BG = "#0f1117"
    PANEL_BG = "#1a1d2e"
    ENTRY_BG = "#252840"
    SEP = "#2d3154"

    # foregrounds
    TEXT = "#e8eaf6"
    SUBTEXT = "#7986cb"

    # accents
    ACCENT = "#00d4ff"  # cyan  – start marker, headings
    ACCENT2 = "#ff6b35"  # orange – end marker

    # typography
    FONT_MONO = "Courier New"
    FONT_MONO_SM = ("Courier New", 8)
    FONT_MONO_MD = ("Courier New", 9)
    FONT_MONO_LG = ("Courier New", 10, "bold")
    FONT_MONO_XL = ("Courier New", 13, "bold")
