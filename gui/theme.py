"""Theme configuration for LogoEraser GUI.

Applies a modern dark palette via ttk.Style and sets Windows immersive dark titlebar.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)

# Dark Theme Color Palette
COLORS = {
    "bg_dark": "#1e1e1e",
    "bg_panel": "#252526",
    "bg_card": "#2d2d30",
    "bg_entry": "#3c3c3c",
    "bg_canvas": "#141414",
    "fg_text": "#e0e0e0",
    "fg_muted": "#888888",
    "fg_accent": "#ffffff",
    "border": "#3f3f46",
    "primary": "#0e639c",
    "primary_hover": "#1177bb",
    "danger": "#b91c1c",
    "danger_hover": "#dc2626",
    "success": "#15803d",
    "warning": "#d97706",
    "select_bg": "#094771",
    "select_fg": "#ffffff",
}


def set_dark_titlebar(window: tk.Tk | tk.Toplevel) -> None:
    """Enable Windows 10/11 immersive dark mode on window titlebar."""
    if sys.platform != "win32":
        return

    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        if not hwnd:
            hwnd = window.winfo_id()

        # DWMWA_USE_IMMERSIVE_DARK_MODE: 20 on Win10 20H1+ and Win11; 19 on earlier builds
        value = ctypes.c_int(1)
        for attr in (20, 19):
            hr = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                attr,
                ctypes.byref(value),
                ctypes.sizeof(value),
            )
            if hr == 0:
                break
    except Exception as e:
        logger.debug("Failed to set dark titlebar: %s", e)


def apply_dark_theme(root: tk.Tk) -> ttk.Style:
    """Configure ttk.Style and root properties for a cohesive dark aesthetic."""
    style = ttk.Style(root)
    # Use 'clam' as base engine for reliable color customization across OS platforms
    try:
        style.theme_use("clam")
    except Exception:
        pass

    root.configure(bg=COLORS["bg_dark"])

    # Configure Base Frame and Labels
    style.configure("TFrame", background=COLORS["bg_dark"])
    style.configure("Panel.TFrame", background=COLORS["bg_panel"])
    style.configure("Card.TFrame", background=COLORS["bg_card"])

    style.configure(
        "TLabel",
        background=COLORS["bg_dark"],
        foreground=COLORS["fg_text"],
        font=("Segoe UI", 9),
    )
    style.configure(
        "Panel.TLabel",
        background=COLORS["bg_panel"],
        foreground=COLORS["fg_text"],
        font=("Segoe UI", 9),
    )
    style.configure(
        "Card.TLabel",
        background=COLORS["bg_card"],
        foreground=COLORS["fg_text"],
        font=("Segoe UI", 9),
    )
    style.configure(
        "Muted.TLabel",
        background=COLORS["bg_dark"],
        foreground=COLORS["fg_muted"],
        font=("Segoe UI", 8),
    )
    style.configure(
        "Header.TLabel",
        background=COLORS["bg_panel"],
        foreground=COLORS["fg_accent"],
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "Warning.TLabel",
        background=COLORS["bg_panel"],
        foreground="#ef4444",
        font=("Segoe UI", 9, "bold"),
    )

    # Configure LabelFrames
    style.configure(
        "TLabelframe",
        background=COLORS["bg_panel"],
        foreground=COLORS["fg_text"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label",
        background=COLORS["bg_panel"],
        foreground=COLORS["fg_text"],
        font=("Segoe UI", 9, "bold"),
    )

    # Configure Buttons
    style.configure(
        "TButton",
        background=COLORS["bg_entry"],
        foreground=COLORS["fg_text"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        relief="flat",
        borderwidth=1,
        padding=(8, 4),
        font=("Segoe UI", 9),
    )
    style.map(
        "TButton",
        background=[("pressed", COLORS["select_bg"]), ("active", "#4a4a4d")],
        foreground=[("disabled", COLORS["fg_muted"])],
    )

    # Primary Action Button (Chạy / Save)
    style.configure(
        "Primary.TButton",
        background=COLORS["primary"],
        foreground=COLORS["fg_accent"],
        relief="flat",
        borderwidth=0,
        padding=(12, 6),
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "Primary.TButton",
        background=[("pressed", "#094771"), ("active", COLORS["primary_hover"])],
        foreground=[("disabled", "#a0a0a0")],
    )

    # Danger Button (Huỷ)
    style.configure(
        "Danger.TButton",
        background=COLORS["danger"],
        foreground=COLORS["fg_accent"],
        relief="flat",
        borderwidth=0,
        padding=(12, 6),
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "Danger.TButton",
        background=[("pressed", "#7f1d1d"), ("active", COLORS["danger_hover"])],
        foreground=[("disabled", "#a0a0a0")],
    )

    # Configure Entries and Comboboxes
    style.configure(
        "TEntry",
        fieldbackground=COLORS["bg_entry"],
        foreground=COLORS["fg_accent"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        insertcolor=COLORS["fg_accent"],
        padding=4,
    )

    style.configure(
        "TCombobox",
        fieldbackground=COLORS["bg_entry"],
        background=COLORS["bg_entry"],
        foreground=COLORS["fg_accent"],
        selectbackground=COLORS["select_bg"],
        selectforeground=COLORS["select_fg"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        arrowcolor=COLORS["fg_text"],
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", COLORS["bg_entry"])],
        selectbackground=[("readonly", COLORS["select_bg"])],
        selectforeground=[("readonly", COLORS["select_fg"])],
    )

    # Configure Checkbuttons & Radiobuttons
    style.configure(
        "TCheckbutton",
        background=COLORS["bg_panel"],
        foreground=COLORS["fg_text"],
        font=("Segoe UI", 9),
    )
    style.map(
        "TCheckbutton",
        background=[("active", COLORS["bg_panel"])],
        foreground=[("disabled", COLORS["fg_muted"])],
    )

    # Configure Progressbar
    style.configure(
        "Horizontal.TProgressbar",
        background=COLORS["primary"],
        troughcolor=COLORS["bg_card"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        thickness=14,
    )

    # Configure Scrollbars
    style.configure(
        "TScrollbar",
        background=COLORS["bg_entry"],
        troughcolor=COLORS["bg_panel"],
        bordercolor=COLORS["border"],
        arrowcolor=COLORS["fg_text"],
        relief="flat",
    )

    # Configure Panedwindow & Notebook
    style.configure("TPanedwindow", background=COLORS["bg_dark"])

    set_dark_titlebar(root)
    return style
