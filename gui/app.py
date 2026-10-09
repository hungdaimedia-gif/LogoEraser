"""Main application window and controller for LogoEraser GUI.

Orchestrates file queue, canvas preview, parameter controls, background worker threads,
queue polling, and safe application termination.
"""

from __future__ import annotations

import copy
import logging
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

from core import __version__
from core.batch import is_clean_output, media_kind, output_path_for, process_file, read_image_safe
from core.config import check_tools, load_config, tool_path
from core.preset import SUPPORTED_METHODS, validate_preset
from core.video_engine import extract_frame
from gui.canvas import ImageCanvas
from gui.state import GuiState
from gui.theme import COLORS

logger = logging.getLogger(__name__)


class LogoEraserApp(ttk.Frame):
    """Main application frame for LogoEraser."""

    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root)
        self.root = root
        self.root.title(f"LogoEraser {__version__} - Công cụ xóa logo & watermark")
        # Hệ số so với 96 DPI: khi đã DPI-aware, mọi kích thước pixel phải nhân lên
        # để cửa sổ không bé tí trên màn 4K.
        self.ui_scale = max(1.0, float(self.root.winfo_fpixels("1i")) / 96.0)
        win_w = min(self._px(1200), int(self.root.winfo_screenwidth() * 0.92))
        win_h = min(self._px(750), int(self.root.winfo_screenheight() * 0.88))
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.minsize(min(self._px(950), win_w), min(self._px(600), win_h))

        self.state = GuiState()
        self.msg_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.cancel_event: threading.Event | None = None
        self.is_processing: bool = False
        self._load_token: int = 0

        self.cfg = load_config()

        self._init_menu()
        self._init_ui()
        self._bind_shortcuts()

        # Handle window close cleanly
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_window)

    def _px(self, n: int) -> int:
        return int(round(n * self.ui_scale))

    def _init_menu(self) -> None:
        """Create menu bar for presets and quick actions."""
        menubar = tk.Menu(self.root, bg=COLORS["bg_panel"], fg=COLORS["fg_text"], activebackground=COLORS["select_bg"])

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=COLORS["bg_panel"], fg=COLORS["fg_text"], activebackground=COLORS["select_bg"])
        file_menu.add_command(label="Thêm tệp tin...", accelerator="Ctrl+O", command=self._on_add_files)
        file_menu.add_command(label="Thêm thư mục...", accelerator="Ctrl+Shift+O", command=self._on_add_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", accelerator="Alt+F4", command=self._on_close_window)
        menubar.add_cascade(label="Tệp tin", menu=file_menu)

        # Preset Menu
        preset_menu = tk.Menu(menubar, tearoff=0, bg=COLORS["bg_panel"], fg=COLORS["fg_text"], activebackground=COLORS["select_bg"])
        preset_menu.add_command(label="Mở Preset (JSON)...", accelerator="Ctrl+L", command=self._on_load_preset)
        preset_menu.add_command(label="Lưu Preset (JSON)...", accelerator="Ctrl+S", command=self._on_save_preset)
        preset_menu.add_separator()
        preset_menu.add_command(label="Đặt lại về mặc định", command=self._on_reset_preset)
        menubar.add_cascade(label="Preset", menu=preset_menu)

        self.root.config(menu=menubar)

    def _init_ui(self) -> None:
        """Create split-pane layout: Files on left, Canvas in center, Parameters on right."""
        self.pack(fill=tk.BOTH, expand=True)

        # Main Paned Window
        self.paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # 1. Left Panel: File list & destination
        self.left_panel = ttk.Frame(self.paned, style="Panel.TFrame", padding=8, width=self._px(250))
        self.paned.add(self.left_panel, weight=1)
        self._build_file_list_panel(self.left_panel)

        # 2. Center Panel: Canvas & Preview Toolbar
        self.center_panel = ttk.Frame(self.paned, style="TFrame", padding=4)
        self.paned.add(self.center_panel, weight=4)
        self._build_canvas_panel(self.center_panel)

        # 3. Right Panel: Method, Parameters, Video Settings, Action
        self.right_panel = ttk.Frame(self.paned, style="Panel.TFrame", padding=8, width=self._px(320))
        self.paned.add(self.right_panel, weight=2)
        self._build_controls_panel(self.right_panel)

        # 4. Bottom Status Bar
        self.status_bar = ttk.Frame(self, style="Card.TFrame", padding=(8, 4))
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_text_var = tk.StringVar(value="Sẵn sàng")
        self.status_lbl = ttk.Label(self.status_bar, textvariable=self.status_text_var, style="Card.TLabel")
        self.status_lbl.pack(side=tk.LEFT)

    def _bind_shortcuts(self) -> None:
        """Bind global keyboard shortcuts."""
        self.root.bind("<Control-o>", lambda e: self._on_add_files())
        self.root.bind("<Control-O>", lambda e: self._on_add_folder())
        self.root.bind("<Control-s>", lambda e: self._on_save_preset())
        self.root.bind("<Control-l>", lambda e: self._on_load_preset())

    # ==================== Left Panel: File List ====================

    def _build_file_list_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="DANH SÁCH TỆP TIN", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 6))

        # Listbox with Scrollbar
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(
            list_frame,
            bg=COLORS["bg_entry"],
            fg=COLORS["fg_text"],
            selectbackground=COLORS["select_bg"],
            selectforeground=COLORS["select_fg"],
            highlightthickness=0,
            borderwidth=1,
            relief="solid",
            font=("Segoe UI", 9),
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_selected)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        # Buttons under listbox
        btn_frame1 = ttk.Frame(parent, style="Panel.TFrame")
        btn_frame1.pack(fill=tk.X, pady=(6, 2))
        ttk.Button(btn_frame1, text="Thêm tệp...", command=self._on_add_files).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_frame1, text="Thêm thư mục...", command=self._on_add_folder).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        btn_frame2 = ttk.Frame(parent, style="Panel.TFrame")
        btn_frame2.pack(fill=tk.X, pady=(2, 8))
        ttk.Button(btn_frame2, text="Xoá khỏi danh sách", command=self._on_remove_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_frame2, text="Xoá tất cả", command=self._on_clear_files).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # Output folder section
        out_frame = ttk.LabelFrame(parent, text="Thư mục xuất", padding=6)
        out_frame.pack(fill=tk.X, pady=(4, 0))

        # Thư mục xuất mặc định lấy từ config.json (trống = cạnh file gốc)
        self.out_dir_var = tk.StringVar(value=str(self.cfg.get("default_output_dir") or ""))
        out_entry = ttk.Entry(out_frame, textvariable=self.out_dir_var)
        out_entry.pack(fill=tk.X, pady=(0, 4))

        out_btn_row = ttk.Frame(out_frame, style="Panel.TFrame")
        out_btn_row.pack(fill=tk.X)
        ttk.Button(out_btn_row, text="Chọn thư mục...", command=self._on_browse_output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_btn_row, text="Mặc định", command=lambda: self.out_dir_var.set("")).pack(side=tk.RIGHT)

        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(out_frame, text="Ghi đè tệp cũ (--overwrite)", variable=self.overwrite_var).pack(anchor=tk.W, pady=(4, 0))

    # ==================== Center Panel: Canvas ====================

    def _build_canvas_panel(self, parent: ttk.Frame) -> None:
        self.canvas_widget = ImageCanvas(parent, self.state, on_boxes_changed=self._on_boxes_changed)
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Toolbar under canvas
        toolbar = ttk.Frame(parent, style="Card.TFrame", padding=6)
        toolbar.pack(fill=tk.X, pady=(4, 0))

        self.btn_preview = ttk.Button(toolbar, text="👁 Xem trước", command=self._on_toggle_preview)
        self.btn_preview.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(toolbar, text="🔍 Vừa màn hình", command=self.canvas_widget.fit_to_window).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="🗑 Xoá vùng chọn (Del)", command=lambda: self.canvas_widget._on_delete_key(None)).pack(side=tk.LEFT)

        self.info_lbl_var = tk.StringVar(value="Chưa có ảnh | 0 vùng logo")
        ttk.Label(toolbar, textvariable=self.info_lbl_var, style="Card.TLabel").pack(side=tk.RIGHT)

    # ==================== Right Panel: Controls & Settings ====================

    def _build_controls_panel(self, parent: ttk.Frame) -> None:
        # 1. Removal Method Group
        method_group = ttk.LabelFrame(parent, text="Phương thức xoá logo", padding=8)
        method_group.pack(fill=tk.X, pady=(0, 6))

        self.method_var = tk.StringVar(value=self.state.method)
        method_combo = self.method_combo = ttk.Combobox(
            method_group,
            textvariable=self.method_var,
            values=list(SUPPORTED_METHODS),
            state="readonly",
        )
        method_combo.pack(fill=tk.X, pady=(0, 4))
        method_combo.bind("<<ComboboxSelected>>", self._on_method_changed)

        # Warning label for clone_smooth on fast mode
        self.warning_lbl_var = tk.StringVar(value="")
        self.warning_lbl = ttk.Label(
            method_group,
            textvariable=self.warning_lbl_var,
            style="Warning.TLabel",
            wraplength=280,
        )

        # 2. Dynamic Parameters Group
        self.params_group = ttk.LabelFrame(parent, text="Tham số phương thức", padding=8)
        self.params_group.pack(fill=tk.X, pady=(0, 6))
        self._build_param_controls(self.params_group)

        # 3. Video Settings Group
        video_group = ttk.LabelFrame(parent, text="Cài đặt Video", padding=8)
        video_group.pack(fill=tk.X, pady=(0, 6))

        # Mode row
        mode_row = ttk.Frame(video_group, style="Panel.TFrame")
        mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text="Chế độ:", style="Panel.TLabel").pack(side=tk.LEFT)
        self.video_mode_var = tk.StringVar(value=self.state.video_mode)
        ttk.Radiobutton(mode_row, text="Fast (FFmpeg)", value="fast", variable=self.video_mode_var, command=self._on_video_mode_changed).pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(mode_row, text="Quality (OpenCV)", value="quality", variable=self.video_mode_var, command=self._on_video_mode_changed).pack(side=tk.LEFT)

        # Encoder row
        enc_row = ttk.Frame(video_group, style="Panel.TFrame")
        enc_row.pack(fill=tk.X, pady=2)
        ttk.Label(enc_row, text="Bộ mã hoá:", style="Panel.TLabel").pack(side=tk.LEFT)
        self.video_enc_var = tk.StringVar(value=self.state.video_encoder)
        enc_combo = ttk.Combobox(
            enc_row,
            textvariable=self.video_enc_var,
            values=["auto", "libx264", "h264_nvenc", "h264_qsv", "h264_amf"],
            state="readonly",
            width=14,
        )
        enc_combo.pack(side=tk.RIGHT)
        enc_combo.bind("<<ComboboxSelected>>", lambda e: self.state.__setattr__("video_encoder", self.video_enc_var.get()))

        # CRF row
        crf_row = ttk.Frame(video_group, style="Panel.TFrame")
        crf_row.pack(fill=tk.X, pady=2)
        ttk.Label(crf_row, text="Chất lượng (CRF 0-51):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.video_crf_var = tk.IntVar(value=self.state.video_crf)
        crf_spin = ttk.Spinbox(crf_row, from_=0, to=51, textvariable=self.video_crf_var, width=5)
        crf_spin.pack(side=tk.RIGHT)
        def on_crf_write(*_: Any) -> None:
            try:
                self.state.video_crf = int(self.video_crf_var.get())
            except (tk.TclError, ValueError):
                pass  # đang gõ dở

        self.video_crf_var.trace_add("write", on_crf_write)

        # Copy audio row
        self.copy_audio_var = tk.BooleanVar(value=self.state.video_copy_audio)
        ttk.Checkbutton(
            video_group,
            text="Giữ nguyên âm thanh",
            variable=self.copy_audio_var,
            command=lambda: self.state.__setattr__("video_copy_audio", self.copy_audio_var.get()),
        ).pack(anchor=tk.W, pady=(4, 0))

        # 4. Action & Progress Group
        action_group = ttk.LabelFrame(parent, text="Thực thi", padding=8)
        action_group.pack(fill=tk.X, pady=(6, 0))

        btn_row = ttk.Frame(action_group, style="Panel.TFrame")
        btn_row.pack(fill=tk.X, pady=(0, 6))

        self.btn_run = ttk.Button(btn_row, text="▶ BẮT ĐẦU XỬ LÝ", style="Primary.TButton", command=self._on_start_processing)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_cancel = ttk.Button(btn_row, text="⏹ HUỶ", style="Danger.TButton", command=self._on_cancel_processing, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.RIGHT, padx=(4, 0))

        self.progress_bar = ttk.Progressbar(action_group, style="Horizontal.TProgressbar", mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(4, 4))

        self.progress_lbl_var = tk.StringVar(value="Đang chờ lệnh...")
        ttk.Label(action_group, textvariable=self.progress_lbl_var, style="Muted.TLabel").pack(anchor=tk.W)

        self._update_param_visibility()

    def _build_param_controls(self, parent: ttk.LabelFrame) -> None:
        """Create controls for all method parameters (dynamically shown/hidden)."""
        self.param_widgets: dict[str, ttk.Frame] = {}

        # 1. Dilate
        f_dilate = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(f_dilate, text="Nới rộng (dilate):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.var_dilate = tk.IntVar(value=self.state.params.get("dilate", 3))
        spin_dilate = ttk.Spinbox(f_dilate, from_=0, to=50, textvariable=self.var_dilate, width=5)
        spin_dilate.pack(side=tk.RIGHT)
        self._bind_param("dilate", self.var_dilate)
        self.param_widgets["dilate"] = f_dilate

        # 2. Feather
        f_feather = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(f_feather, text="Làm mềm biên (feather):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.var_feather = tk.IntVar(value=self.state.params.get("feather", 4))
        spin_feather = ttk.Spinbox(f_feather, from_=0, to=50, textvariable=self.var_feather, width=5)
        spin_feather.pack(side=tk.RIGHT)
        self._bind_param("feather", self.var_feather)
        self.param_widgets["feather"] = f_feather

        # 3. Inpaint radius
        f_radius = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(f_radius, text="Bán kính inpaint (radius):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.var_radius = tk.IntVar(value=self.state.params.get("radius", 5))
        spin_radius = ttk.Spinbox(f_radius, from_=1, to=50, textvariable=self.var_radius, width=5)
        spin_radius.pack(side=tk.RIGHT)
        self._bind_param("radius", self.var_radius)
        self.param_widgets["radius"] = f_radius

        # 4. Blur ksize
        f_ksize = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(f_ksize, text="Độ mờ blur (ksize lẻ):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.var_ksize = tk.IntVar(value=self.state.params.get("blur_ksize", 21))
        spin_ksize = ttk.Spinbox(f_ksize, from_=3, to=99, increment=2, textvariable=self.var_ksize, width=5)
        spin_ksize.pack(side=tk.RIGHT)
        self._bind_param("blur_ksize", self.var_ksize)
        self.param_widgets["blur_ksize"] = f_ksize

        # 5. Mosaic block
        f_mosaic = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(f_mosaic, text="Kích thước ô mosaic:", style="Panel.TLabel").pack(side=tk.LEFT)
        self.var_mosaic = tk.IntVar(value=self.state.params.get("mosaic_block", 12))
        spin_mosaic = ttk.Spinbox(f_mosaic, from_=2, to=64, textvariable=self.var_mosaic, width=5)
        spin_mosaic.pack(side=tk.RIGHT)
        self._bind_param("mosaic_block", self.var_mosaic)
        self.param_widgets["mosaic_block"] = f_mosaic

        # 6. Solid color
        f_color = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(f_color, text="Màu đè (Hex):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.var_color = tk.StringVar(value=self.state.params.get("solid_color", "#000000"))
        ttk.Button(f_color, text="🎨 Chọn...", command=self._on_choose_color, width=8).pack(side=tk.RIGHT, padx=(4, 0))
        entry_color = ttk.Entry(f_color, textvariable=self.var_color, width=9)
        entry_color.pack(side=tk.RIGHT)
        self._bind_param("solid_color", self.var_color)
        self.param_widgets["solid_color"] = f_color

        # 7. Overlay path
        f_ov_path = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(f_ov_path, text="Ảnh phủ (PNG):", style="Panel.TLabel").pack(anchor=tk.W)
        row_ov = ttk.Frame(f_ov_path, style="Panel.TFrame")
        row_ov.pack(fill=tk.X, pady=(2, 0))
        self.var_ov_path = tk.StringVar(value=self.state.params.get("overlay_path", ""))
        entry_ov = ttk.Entry(row_ov, textvariable=self.var_ov_path)
        entry_ov.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._bind_param("overlay_path", self.var_ov_path)
        ttk.Button(row_ov, text="Chọn...", command=self._on_choose_overlay).pack(side=tk.RIGHT)
        self.param_widgets["overlay_path"] = f_ov_path

        # 8. Overlay opacity
        f_ov_op = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Label(f_ov_op, text="Độ mờ phủ (0.0-1.0):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.var_ov_op = tk.DoubleVar(value=self.state.params.get("overlay_opacity", 1.0))
        spin_ov_op = ttk.Spinbox(f_ov_op, from_=0.0, to=1.0, increment=0.05, textvariable=self.var_ov_op, width=5)
        spin_ov_op.pack(side=tk.RIGHT)
        self._bind_param("overlay_opacity", self.var_ov_op)
        self.param_widgets["overlay_opacity"] = f_ov_op

    def _update_param_visibility(self) -> None:
        """Show only relevant parameter controls for the current method (Review 1b-r2 Note 4)."""
        active_keys = set(self.state.get_active_param_names())

        for key, frame in self.param_widgets.items():
            if key in active_keys:
                frame.pack(fill=tk.X, pady=3)
            else:
                frame.pack_forget()

        # Warning banner for clone_smooth + fast video mode
        if self.state.method == "clone_smooth" and self.video_mode_var.get() == "fast":
            self.warning_lbl_var.set("⚠ clone_smooth không hỗ trợ video chế độ fast. Vui lòng chuyển sang quality!")
            self.warning_lbl.pack(fill=tk.X, pady=(4, 0))
        else:
            self.warning_lbl_var.set("")
            self.warning_lbl.pack_forget()

    # ==================== Event Callbacks ====================

    def _on_method_changed(self, event: tk.Event | None = None) -> None:
        new_method = self.method_var.get()
        self.state.set_method(new_method)
        self._update_param_visibility()
        if self.canvas_widget.show_preview:
            self.canvas_widget.request_preview()

    def _bind_param(self, key: str, var: tk.Variable) -> None:
        """Đẩy giá trị vào state ngay khi biến đổi (gõ, bấm mũi tên Spinbox, chọn màu).

        Trước đây chỉ bắt <FocusOut>: bấm mũi tên rồi bấm Chạy ngay thì giá trị mới
        không vào preset. Giá trị dở dang khi đang gõ (ô trống, "0.") thì bỏ qua.
        """

        def on_write(*_: Any) -> None:
            try:
                value = var.get()
            except (tk.TclError, ValueError):
                return
            self._on_param_changed(key, value)

        var.trace_add("write", on_write)

    def _on_param_changed(self, key: str, value: Any) -> None:
        self.state.set_param(key, value)
        if self.canvas_widget.show_preview:
            self.canvas_widget.request_preview()

    def _on_video_mode_changed(self) -> None:
        mode = self.video_mode_var.get()
        self.state.set_video_mode(mode)
        self._update_param_visibility()

    def _on_boxes_changed(self) -> None:
        num_boxes = len(self.state.boxes)
        dims = f"{self.canvas_widget.image_width}×{self.canvas_widget.image_height}" if self.canvas_widget.image_width > 0 else "Chưa có ảnh"
        zoom_pct = int(round(self.canvas_widget.zoom * 100))
        self.info_lbl_var.set(f"{dims} | {num_boxes} vùng logo | Zoom: {zoom_pct}%")

        if self.canvas_widget.show_preview:
            self.canvas_widget.request_preview()

    def _on_toggle_preview(self) -> None:
        showing = self.canvas_widget.toggle_preview()
        self.btn_preview.config(text="👁 Ẩn xem trước" if showing else "👁 Xem trước")

    def _on_choose_color(self) -> None:
        color = colorchooser.askcolor(title="Chọn màu đè")[1]
        if color:
            self.var_color.set(color)
            self._on_param_changed("solid_color", color)

    def _on_choose_overlay(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn ảnh phủ (PNG RGBA)",
            filetypes=[("PNG Image", "*.png"), ("All files", "*.*")],
        )
        if path:
            self.var_ov_path.set(path)
            self._on_param_changed("overlay_path", path)

    # ==================== File List Operations ====================

    def _on_add_files(self) -> None:
        file_types = [
            ("Tệp đa phương tiện", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.mp4;*.mov;*.mkv;*.webm;*.avi"),
            ("Hình ảnh", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"),
            ("Video", "*.mp4;*.mov;*.mkv;*.webm;*.avi"),
            ("Tất cả tệp", "*.*"),
        ]
        chosen = filedialog.askopenfilenames(title="Chọn ảnh hoặc video", filetypes=file_types)
        if chosen:
            added = self.state.add_files(list(chosen))
            self._refresh_file_listbox()
            if added > 0 and self.canvas_widget.original_image is None:
                self._load_file_to_canvas(self.state.files[0])

    def _on_add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh/video")
        if folder:
            paths: list[str] = []
            for root, _, files in os.walk(folder):
                for f in files:
                    # Bỏ qua kết quả của lần chạy trước (*_clean.*), giống CLI (N15).
                    if not is_clean_output(f):
                        paths.append(os.path.join(root, f))
            added = self.state.add_files(paths)
            self._refresh_file_listbox()
            if added > 0 and self.canvas_widget.original_image is None:
                self._load_file_to_canvas(self.state.files[0])

    def _on_remove_file(self) -> None:
        sel = self.file_listbox.curselection()
        if sel:
            idx = sel[0]
            self.state.remove_file(idx)
            self._refresh_file_listbox()
            if self.state.selected_file_idx is not None:
                self._load_file_to_canvas(self.state.files[self.state.selected_file_idx])
            else:
                self.canvas_widget.set_image(None)

    def _on_clear_files(self) -> None:
        self.state.clear_files()
        self._refresh_file_listbox()
        self.canvas_widget.set_image(None)

    def _on_file_selected(self, event: tk.Event) -> None:
        sel = self.file_listbox.curselection()
        if sel:
            idx = sel[0]
            file_path = self.state.select_file(idx)
            if file_path:
                self._load_file_to_canvas(file_path)

    def _refresh_file_listbox(self) -> None:
        self.file_listbox.delete(0, tk.END)
        for f in self.state.files:
            self.file_listbox.insert(tk.END, os.path.basename(f))
        if self.state.selected_file_idx is not None:
            self.file_listbox.select_set(self.state.selected_file_idx)

    def _load_file_to_canvas(self, path: str) -> None:
        """Load image or first video frame to display on Canvas.

        Khung đầu của video đọc bằng ffprobe + ffmpeg trong luồng phụ: gọi
        subprocess trên luồng Tk làm đơ giao diện với video lớn (REVIEW-2 N18).
        """
        kind = media_kind(path)
        if kind is None:
            messagebox.showerror("Lỗi mở tệp", f"Định dạng không hỗ trợ: '{path}'")
            return

        self._load_token += 1
        token = self._load_token

        if kind == "image":
            try:
                img = read_image_safe(path)
            except Exception as e:
                messagebox.showerror("Lỗi mở tệp", f"Không thể đọc tệp '{path}':\n{e}")
                return
            self._show_loaded_frame(path, img)
            return

        self.status_text_var.set(f"Đang đọc khung đầu: {os.path.basename(path)}...")
        ffmpeg_bin = tool_path(self.cfg, "ffmpeg")
        ffprobe_bin = tool_path(self.cfg, "ffprobe")
        tool_err = check_tools(ffmpeg_bin, ffprobe_bin)
        if tool_err:
            self.status_text_var.set("Sẵn sàng")
            messagebox.showerror("Thiếu FFmpeg", tool_err)
            return
        result: dict[str, Any] = {}

        def work() -> None:
            try:
                result["frame"] = extract_frame(path, timestamp_sec=0.0, ffmpeg=ffmpeg_bin, ffprobe=ffprobe_bin)
            except Exception as e:  # báo lại trên luồng Tk
                result["error"] = e

        t = threading.Thread(target=work, daemon=True)
        t.start()

        def poll() -> None:
            if t.is_alive():
                self.root.after(50, poll)
                return
            if token != self._load_token:
                return  # người dùng đã chọn file khác trong lúc chờ
            if "error" in result:
                self.status_text_var.set("Sẵn sàng")
                messagebox.showerror("Lỗi mở tệp", f"Không thể đọc tệp '{path}':\n{result['error']}")
                return
            self._show_loaded_frame(path, result["frame"])

        self.root.after(50, poll)

    def _show_loaded_frame(self, path: str, img: np.ndarray) -> None:
        self.canvas_widget.set_image(img)
        self._on_boxes_changed()
        self.status_text_var.set(f"Đã mở: {os.path.basename(path)}")

    def _on_browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục xuất kết quả")
        if folder:
            self.out_dir_var.set(folder)

    # ==================== Preset Management ====================

    def _on_save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Lưu cấu hình Preset",
            defaultextension=".json",
            filetypes=[("JSON Preset", "*.json"), ("All files", "*.*")],
        )
        if path:
            try:
                self.state.save_preset_file(path)
                self.status_text_var.set(f"Đã lưu preset: {os.path.basename(path)}")
                messagebox.showinfo("Thành công", f"Đã lưu preset thành công vào:\n{path}")
            except Exception as e:
                messagebox.showerror("Lỗi lưu preset", f"Không thể lưu preset:\n{e}")

    def _on_load_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Mở cấu hình Preset",
            filetypes=[("JSON Preset", "*.json"), ("All files", "*.*")],
        )
        if path:
            try:
                self.state.load_preset_file(path)
                # Update GUI widgets with new state
                self.method_var.set(self.state.method)
                self.video_mode_var.set(self.state.video_mode)
                self.video_enc_var.set(self.state.video_encoder)
                self.video_crf_var.set(self.state.video_crf)
                self.copy_audio_var.set(self.state.video_copy_audio)

                for k, v in self.state.params.items():
                    if k == "dilate":
                        self.var_dilate.set(int(v))
                    elif k == "feather":
                        self.var_feather.set(int(v))
                    elif k == "radius":
                        self.var_radius.set(int(v))
                    elif k == "blur_ksize":
                        self.var_ksize.set(int(v))
                    elif k == "mosaic_block":
                        self.var_mosaic.set(int(v))
                    elif k == "solid_color":
                        self.var_color.set(str(v))
                    elif k == "overlay_path":
                        self.var_ov_path.set(str(v))
                    elif k == "overlay_opacity":
                        self.var_ov_op.set(float(v))

                self._update_param_visibility()
                self.canvas_widget.redraw()
                self._on_boxes_changed()
                self.status_text_var.set(f"Đã nạp preset: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Lỗi mở preset", f"Không thể đọc preset:\n{e}")

    def _on_reset_preset(self) -> None:
        # Chỉ đặt lại preset — giữ danh sách file đang mở (trước đây state.__init__()
        # xoá luôn danh sách file khỏi giao diện).
        files, selected = list(self.state.files), self.state.selected_file_idx
        self.state.__init__()
        self.state.files, self.state.selected_file_idx = files, selected
        self.method_var.set(self.state.method)
        self.video_mode_var.set(self.state.video_mode)
        self.video_enc_var.set(self.state.video_encoder)
        self.video_crf_var.set(self.state.video_crf)
        self.copy_audio_var.set(self.state.video_copy_audio)
        p = self.state.params
        self.var_dilate.set(p["dilate"])
        self.var_feather.set(p["feather"])
        self.var_radius.set(p["radius"])
        self.var_ksize.set(p["blur_ksize"])
        self.var_mosaic.set(p["mosaic_block"])
        self.var_color.set(p["solid_color"])
        self.var_ov_path.set(p["overlay_path"] or "")
        self.var_ov_op.set(p["overlay_opacity"])
        self._update_param_visibility()
        self.canvas_widget.redraw()
        self._on_boxes_changed()

    # ==================== Background Worker & Execution ====================

    def _on_start_processing(self) -> None:
        """Validate state, disable controls, and launch background processing thread."""
        errors = self.state.validate_for_run()
        if errors:
            messagebox.showerror("Không thể bắt đầu", "\n• " + "\n• ".join(errors))
            return

        preset = self.state.to_preset_dict()
        files = list(self.state.files)
        output_dir = self.out_dir_var.get().strip()
        overwrite = self.overwrite_var.get()
        ffmpeg_bin = tool_path(self.cfg, "ffmpeg")
        ffprobe_bin = tool_path(self.cfg, "ffprobe")

        if any(media_kind(f) == "video" for f in files):
            tool_err = check_tools(ffmpeg_bin, ffprobe_bin)
            if tool_err:
                messagebox.showerror("Thiếu FFmpeg", tool_err)
                return

        self.cancel_event = threading.Event()
        self.is_processing = True

        self.btn_run.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.progress_bar["value"] = 0
        self.progress_lbl_var.set("Đang chuẩn bị...")
        self.status_text_var.set("Đang xử lý...")

        self.worker_thread = threading.Thread(
            target=self._worker_run,
            args=(files, preset, output_dir, overwrite, ffmpeg_bin, ffprobe_bin, self.cancel_event),
            daemon=True,
        )
        self.worker_thread.start()
        self.root.after(50, self._poll_queue)

    def _worker_run(
        self,
        files: list[str],
        preset: dict[str, Any],
        output_dir: str,
        overwrite: bool,
        ffmpeg_bin: str,
        ffprobe_bin: str,
        cancel_evt: threading.Event,
    ) -> None:
        """Execute processing in worker thread, pushing progress to msg_queue."""
        total_files = len(files)
        success_count = 0
        failures: list[tuple[str, str]] = []

        for idx, src in enumerate(files):
            if cancel_evt.is_set():
                break

            dst = output_path_for(src, output_dir or None)
            self.msg_queue.put(("file_start", idx + 1, total_files, src))

            def on_prog(cur: float, total: float, _idx: int = idx) -> None:
                pct = (cur / total * 100.0) if total > 0 else 0.0
                self.msg_queue.put(("video_progress", _idx + 1, total_files, pct))

            try:
                process_file(
                    src,
                    dst,
                    preset,
                    overwrite=overwrite,
                    progress_cb=on_prog,
                    cancel=cancel_evt,
                    ffmpeg=ffmpeg_bin,
                    ffprobe=ffprobe_bin,
                )
                success_count += 1
                self.msg_queue.put(("file_done", idx + 1, total_files, dst))
            except Exception as e:
                if cancel_evt.is_set():
                    break
                failures.append((src, str(e)))
                self.msg_queue.put(("file_failed", idx + 1, total_files, src, str(e)))

        if cancel_evt.is_set():
            self.msg_queue.put(("cancelled", success_count, failures))
        else:
            self.msg_queue.put(("all_done", success_count, failures))

    def _poll_queue(self) -> None:
        """Poll worker messages on Tk main thread every 50ms."""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                tag = msg[0]

                if tag == "file_start":
                    _, cur_idx, total_cnt, fpath = msg
                    base = os.path.basename(fpath)
                    self.progress_lbl_var.set(f"Đang xử lý ({cur_idx}/{total_cnt}): {base}")
                    pct = ((cur_idx - 1) / total_cnt) * 100
                    self.progress_bar["value"] = pct

                elif tag == "video_progress":
                    _, cur_idx, total_cnt, vid_pct = msg
                    base_pct = ((cur_idx - 1) / total_cnt) * 100
                    item_pct = (vid_pct / total_cnt)
                    self.progress_bar["value"] = base_pct + item_pct
                    self.progress_lbl_var.set(f"Đang xử lý tệp ({cur_idx}/{total_cnt}): {vid_pct:.1f}%")

                elif tag == "file_done":
                    _, cur_idx, total_cnt, dst = msg
                    self.progress_bar["value"] = (cur_idx / total_cnt) * 100

                elif tag == "all_done":
                    _, success_cnt, fails = msg
                    self._on_processing_completed(success_cnt, fails)
                    return

                elif tag == "cancelled":
                    _, success_cnt, fails = msg
                    self._on_processing_cancelled(success_cnt)
                    return

        except queue.Empty:
            pass

        if self.is_processing:
            self.root.after(50, self._poll_queue)

    def _on_processing_completed(self, success_cnt: int, fails: list[tuple[str, str]]) -> None:
        self.is_processing = False
        self.btn_run.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self.progress_bar["value"] = 100

        if not fails:
            self.status_text_var.set(f"Hoàn thành xuất sắc: {success_cnt} tệp")
            self.progress_lbl_var.set(f"Đã hoàn thành toàn bộ {success_cnt} tệp.")
            messagebox.showinfo("Hoàn tất", f"Đã xử lý thành công tất cả {success_cnt} tệp tin!")
        else:
            self.status_text_var.set(f"Hoàn thành với lỗi: {success_cnt} thành công, {len(fails)} lỗi")
            fail_msg = "\n".join(f"• {os.path.basename(f)}: {err}" for f, err in fails[:5])
            if len(fails) > 5:
                fail_msg += f"\n... và {len(fails) - 5} lỗi khác."
            messagebox.showwarning("Hoàn tất có lỗi", f"Xử lý xong {success_cnt} tệp.\nLỗi ở {len(fails)} tệp:\n\n{fail_msg}")

    def _on_processing_cancelled(self, success_cnt: int) -> None:
        self.is_processing = False
        if getattr(self, "_closing", False):
            return
        self.btn_run.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self.status_text_var.set("Đã dừng tiến trình theo yêu cầu")
        self.progress_lbl_var.set("Tiến trình đã bị huỷ.")
        messagebox.showinfo("Đã huỷ", f"Đã dừng tiến trình. Số tệp hoàn tất trước khi dừng: {success_cnt}")

    def _on_cancel_processing(self) -> None:
        if self.is_processing and self.cancel_event:
            self.cancel_event.set()
            self.btn_cancel.config(state=tk.DISABLED)
            self.status_text_var.set("Đang yêu cầu dừng tiến trình...")

    def _on_close_window(self) -> None:
        """Handle window close: cancel job, wait until the worker has really stopped, then exit.

        Không đặt trần thời gian chờ: process_video cần thời gian terminate → kill
        tiến trình ffmpeg; destroy() sớm sẽ để lại ffmpeg mồ côi (REVIEW-2 N20).
        """
        if not self.is_processing:
            self.root.destroy()
            return
        if not messagebox.askyesno("Đang xử lý", "Tiến trình xử lý đang chạy. Bạn có chắc chắn muốn huỷ và thoát không?"):
            return
        if self.cancel_event:
            self.cancel_event.set()
        self.btn_cancel.config(state=tk.DISABLED)
        self.btn_run.config(state=tk.DISABLED)
        self.status_text_var.set("Đang dừng tiến trình ffmpeg, cửa sổ sẽ tự đóng...")
        self._closing = True
        self._wait_worker_then_destroy()

    def _wait_worker_then_destroy(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.root.after(100, self._wait_worker_then_destroy)
            return
        self.root.destroy()
