"""Interactive canvas widget for LogoEraser GUI.

Supports image/video frame display, wheel zooming centered at mouse pointer,
middle-mouse/space panning, box creation, 8 resize handles, and debounced preview.
"""

from __future__ import annotations

import copy
import logging
import math
import tkinter as tk
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.image_engine import apply
from core.preset import to_abs_box, to_rel_box
from gui.state import GuiState

logger = logging.getLogger(__name__)

HANDLE_SIZE = 8
HANDLE_NAMES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")


class ImageCanvas(tk.Frame):
    """Canvas widget for displaying images, drawing, resizing, and previewing watermark removal."""

    def __init__(
        self,
        master: tk.Widget,
        state: GuiState,
        on_boxes_changed: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self.state = state
        self.on_boxes_changed = on_boxes_changed

        self.canvas = tk.Canvas(
            self,
            bg="#141414",
            highlightthickness=0,
            cursor="cross",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Image state
        self.original_image: np.ndarray | None = None  # BGR
        self.preview_image: np.ndarray | None = None  # BGR
        self.show_preview: bool = False
        self.image_width: int = 0
        self.image_height: int = 0

        # Viewport transform: canvas_pt = img_pt * zoom + pan
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        # Interaction state
        self.space_pressed: bool = False
        self.drag_mode: str | None = None  # 'pan', 'draw', 'move', 'resize'
        self.drag_start_canvas: tuple[float, float] = (0.0, 0.0)
        self.drag_start_img: tuple[float, float] = (0.0, 0.0)
        self.active_handle: str | None = None
        self.orig_box_on_drag: dict[str, Any] | None = None
        self.temp_draw_box: tuple[float, float, float, float] | None = None  # (x1, y1, x2, y2) in image coords

        # Cached photo image reference to prevent GC
        self._photo_image: ImageTk.PhotoImage | None = None

        # Debounce timer for preview computation (150ms per PLAN §7)
        self._preview_timer_id: str | None = None

        self._bind_events()

    def _bind_events(self) -> None:
        """Bind keyboard and mouse events."""
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_mousewheel(e, 120))  # Linux zoom in
        self.canvas.bind("<Button-5>", lambda e: self._on_mousewheel(e, -120))  # Linux zoom out

        # Pan via middle mouse button
        self.canvas.bind("<ButtonPress-2>", self._on_middle_down)
        self.canvas.bind("<B2-Motion>", self._on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_middle_up)

        # Primary mouse button
        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Motion>", self._on_mouse_move)

        # Keyboard shortcuts
        self.canvas.bind("<KeyPress-space>", self._on_space_down)
        self.canvas.bind("<KeyRelease-space>", self._on_space_up)
        self.canvas.bind("<Delete>", self._on_delete_key)
        self.canvas.bind("<BackSpace>", self._on_delete_key)

        # Ensure canvas can receive keyboard events
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

    # ---------------- Image Loading & Viewport ----------------

    def set_image(self, bgr_img: np.ndarray | None, reset_view: bool = True) -> None:
        """Set the displayed image (BGR numpy array)."""
        self.original_image = bgr_img
        self.preview_image = None
        self.show_preview = False

        if bgr_img is not None:
            self.image_height, self.image_width = bgr_img.shape[:2]
            if reset_view:
                self.fit_to_window()
            else:
                self.redraw()
        else:
            self.image_width = 0
            self.image_height = 0
            self.canvas.delete("all")

    def fit_to_window(self) -> None:
        """Fit the image into the canvas viewport with a small padding margin."""
        if self.original_image is None or self.image_width <= 0 or self.image_height <= 0:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            # Not yet rendered, retry after 50ms
            self.after(50, self.fit_to_window)
            return

        margin = 20
        scale_x = (canvas_w - margin * 2) / self.image_width
        scale_y = (canvas_h - margin * 2) / self.image_height
        self.zoom = min(scale_x, scale_y, 4.0)
        self.zoom = max(self.zoom, 0.05)

        # Center in canvas
        self.pan_x = (canvas_w - self.image_width * self.zoom) / 2.0
        self.pan_y = (canvas_h - self.image_height * self.zoom) / 2.0
        self.redraw()

    def img_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        """Convert image coordinates to canvas coordinates."""
        return ix * self.zoom + self.pan_x, iy * self.zoom + self.pan_y

    def canvas_to_img(self, cx: float, cy: float) -> tuple[float, float]:
        """Convert canvas coordinates to image coordinates."""
        return (cx - self.pan_x) / self.zoom, (cy - self.pan_y) / self.zoom

    # ---------------- Drawing & Rendering ----------------

    def redraw(self) -> None:
        """Redraw image, boxes, handles, and labels on canvas."""
        self.canvas.delete("all")
        if self.original_image is None or self.image_width <= 0 or self.image_height <= 0:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="Chưa chọn tệp ảnh hoặc video\n(Nhấn 'Thêm tệp...' hoặc Ctrl+O để bắt đầu)",
                fill="#666666",
                font=("Segoe UI", 12),
                justify=tk.CENTER,
            )
            return

        # 1. Render image (either preview or original)
        active_img = self.preview_image if (self.show_preview and self.preview_image is not None) else self.original_image
        disp_w = max(1, int(round(self.image_width * self.zoom)))
        disp_h = max(1, int(round(self.image_height * self.zoom)))

        # Convert BGR to RGB for PIL
        rgb = cv2.cvtColor(active_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        if disp_w != self.image_width or disp_h != self.image_height:
            # High quality resampling
            pil_img = pil_img.resize((disp_w, disp_h), Image.Resampling.BILINEAR)

        self._photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(
            self.pan_x,
            self.pan_y,
            anchor=tk.NW,
            image=self._photo_image,
            tags="background_image",
        )

        # 2. Render bounding boxes
        abs_boxes = self.state.get_abs_boxes(self.image_width, self.image_height)
        for idx, box in enumerate(abs_boxes):
            bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
            x1, y1 = self.img_to_canvas(bx, by)
            x2, y2 = self.img_to_canvas(bx + bw, by + bh)

            is_selected = idx == self.state.selected_box_idx
            line_color = "#38bdf8" if is_selected else "#f59e0b"
            line_width = 2 if is_selected else 1

            # Semi-transparent dashed rectangle
            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=line_color,
                width=line_width,
                dash=(4, 3) if not is_selected else (),
                tags=("box", f"box_{idx}"),
            )

            # Box number badge
            badge_text = f"#{idx + 1}"
            self.canvas.create_rectangle(
                x1,
                y1 - 18,
                x1 + 24,
                y1,
                fill=line_color,
                outline="",
                tags=("badge", f"badge_{idx}"),
            )
            self.canvas.create_text(
                x1 + 12,
                y1 - 9,
                text=badge_text,
                fill="#000000",
                font=("Segoe UI", 8, "bold"),
                tags=("badge_text", f"badge_text_{idx}"),
            )

            # If selected, render 8 resize handles
            if is_selected:
                self._draw_handles(x1, y1, x2, y2)

        # 3. Render temporary drag box during creation
        if self.temp_draw_box is not None:
            tx1, ty1, tx2, ty2 = self.temp_draw_box
            cx1, cy1 = self.img_to_canvas(tx1, ty1)
            cx2, cy2 = self.img_to_canvas(tx2, ty2)
            self.canvas.create_rectangle(
                cx1,
                cy1,
                cx2,
                cy2,
                outline="#22c55e",
                width=2,
                dash=(3, 2),
                tags="temp_box",
            )

    def _draw_handles(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Draw 8 resize handles around the selected box."""
        h_half = HANDLE_SIZE / 2
        positions = {
            "nw": (x1, y1),
            "n": ((x1 + x2) / 2, y1),
            "ne": (x2, y1),
            "e": (x2, (y1 + y2) / 2),
            "se": (x2, y2),
            "s": ((x1 + x2) / 2, y2),
            "sw": (x1, y2),
            "w": (x1, (y1 + y2) / 2),
        }

        for name, (hx, hy) in positions.items():
            self.canvas.create_rectangle(
                hx - h_half,
                hy - h_half,
                hx + h_half,
                hy + h_half,
                fill="#ffffff",
                outline="#0284c7",
                width=1.5,
                tags=("handle", f"handle_{name}"),
            )

    # ---------------- Mouse & Keyboard Event Handlers ----------------

    def _on_resize(self, event: tk.Event) -> None:
        self.redraw()

    def _on_mousewheel(self, event: tk.Event, custom_delta: int | None = None) -> None:
        """Zoom centered at mouse cursor position."""
        if self.original_image is None or self.image_width <= 0:
            return

        delta = custom_delta if custom_delta is not None else event.delta
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(0.05, min(20.0, self.zoom * factor))

        # Mouse coordinate on canvas
        mx, my = event.x, event.y
        # Corresponding image coordinate
        ix = (mx - self.pan_x) / self.zoom
        iy = (my - self.pan_y) / self.zoom

        # Adjust pan so (ix, iy) stays invariant at (mx, my)
        self.pan_x = mx - ix * new_zoom
        self.pan_y = my - iy * new_zoom
        self.zoom = new_zoom
        self.redraw()

    def _on_space_down(self, event: tk.Event) -> None:
        self.space_pressed = True
        self.canvas.config(cursor="fleur")

    def _on_space_up(self, event: tk.Event) -> None:
        self.space_pressed = False
        self.canvas.config(cursor="cross")

    def _on_middle_down(self, event: tk.Event) -> None:
        self.drag_mode = "pan"
        self.drag_start_canvas = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _on_middle_drag(self, event: tk.Event) -> None:
        if self.drag_mode == "pan":
            dx = event.x - self.drag_start_canvas[0]
            dy = event.y - self.drag_start_canvas[1]
            self.pan_x += dx
            self.pan_y += dy
            self.drag_start_canvas = (event.x, event.y)
            self.redraw()

    def _on_middle_up(self, event: tk.Event) -> None:
        self.drag_mode = None
        self.canvas.config(cursor="cross" if not self.space_pressed else "fleur")

    def _on_left_down(self, event: tk.Event) -> None:
        """Handle mouse press: Pan, Resize handle, Move box, or Draw new box."""
        if self.original_image is None:
            return

        if self.space_pressed:
            self.drag_mode = "pan"
            self.drag_start_canvas = (event.x, event.y)
            return

        cx, cy = event.x, event.y
        ix, iy = self.canvas_to_img(cx, cy)
        self.drag_start_canvas = (cx, cy)
        self.drag_start_img = (ix, iy)

        # 1. Check if clicked on a resize handle of selected box
        handle = self._find_handle_at(cx, cy)
        if handle and self.state.selected_box_idx is not None:
            self.drag_mode = "resize"
            self.active_handle = handle
            sel_idx = self.state.selected_box_idx
            abs_boxes = self.state.get_abs_boxes(self.image_width, self.image_height)
            self.orig_box_on_drag = copy.deepcopy(abs_boxes[sel_idx])
            return

        # 2. Check if clicked inside any existing box
        box_idx = self._find_box_at(ix, iy)
        if box_idx is not None:
            self.state.selected_box_idx = box_idx
            self.drag_mode = "move"
            abs_boxes = self.state.get_abs_boxes(self.image_width, self.image_height)
            self.orig_box_on_drag = copy.deepcopy(abs_boxes[box_idx])
            self.redraw()
            if self.on_boxes_changed:
                self.on_boxes_changed()
            return

        # 3. Clicked on empty space: start drawing new box
        self.state.selected_box_idx = None
        self.drag_mode = "draw"
        self.temp_draw_box = (ix, iy, ix, iy)
        self.redraw()
        if self.on_boxes_changed:
            self.on_boxes_changed()

    def _on_left_drag(self, event: tk.Event) -> None:
        """Handle left mouse drag."""
        if not self.drag_mode:
            return

        cx, cy = event.x, event.y
        ix, iy = self.canvas_to_img(cx, cy)

        if self.drag_mode == "pan":
            dx = cx - self.drag_start_canvas[0]
            dy = cy - self.drag_start_canvas[1]
            self.pan_x += dx
            self.pan_y += dy
            self.drag_start_canvas = (cx, cy)
            self.redraw()

        elif self.drag_mode == "draw":
            start_ix, start_iy = self.drag_start_img
            x1 = max(0.0, min(self.image_width, min(start_ix, ix)))
            y1 = max(0.0, min(self.image_height, min(start_iy, iy)))
            x2 = max(0.0, min(self.image_width, max(start_ix, ix)))
            y2 = max(0.0, min(self.image_height, max(start_iy, iy)))
            self.temp_draw_box = (x1, y1, x2, y2)
            self.redraw()

        elif self.drag_mode == "move" and self.orig_box_on_drag and self.state.selected_box_idx is not None:
            dx = ix - self.drag_start_img[0]
            dy = iy - self.drag_start_img[1]
            orig = self.orig_box_on_drag
            new_x = int(round(max(0, min(self.image_width - orig["w"], orig["x"] + dx))))
            new_y = int(round(max(0, min(self.image_height - orig["h"], orig["y"] + dy))))
            self.state.update_box_abs(
                self.state.selected_box_idx,
                new_x,
                new_y,
                orig["w"],
                orig["h"],
                self.image_width,
                self.image_height,
            )
            self.redraw()

        elif self.drag_mode == "resize" and self.orig_box_on_drag and self.state.selected_box_idx is not None:
            self._handle_resize_drag(ix, iy)
            self.redraw()

    def _handle_resize_drag(self, cur_ix: float, cur_iy: float) -> None:
        """Compute new box dimensions based on the active resize handle."""
        assert self.orig_box_on_drag is not None
        assert self.active_handle is not None
        sel_idx = self.state.selected_box_idx
        assert sel_idx is not None

        ox = self.orig_box_on_drag["x"]
        oy = self.orig_box_on_drag["y"]
        ow = self.orig_box_on_drag["w"]
        oh = self.orig_box_on_drag["h"]
        ox2 = ox + ow
        oy2 = oy + oh

        h = self.active_handle
        if "w" in h:
            ox = min(cur_ix, ox2 - 4)
        if "e" in h:
            ox2 = max(cur_ix, ox + 4)
        if "n" in h:
            oy = min(cur_iy, oy2 - 4)
        if "s" in h:
            oy2 = max(cur_iy, oy + 4)

        ox = max(0, min(self.image_width - 4, int(round(ox))))
        oy = max(0, min(self.image_height - 4, int(round(oy))))
        ox2 = max(ox + 4, min(self.image_width, int(round(ox2))))
        oy2 = max(oy + 4, min(self.image_height, int(round(oy2))))

        self.state.update_box_abs(
            sel_idx,
            ox,
            oy,
            ox2 - ox,
            oy2 - oy,
            self.image_width,
            self.image_height,
        )

    def _on_left_up(self, event: tk.Event) -> None:
        """Finalize mouse action on release."""
        if self.drag_mode == "draw" and self.temp_draw_box is not None:
            x1, y1, x2, y2 = self.temp_draw_box
            w = int(round(x2 - x1))
            h = int(round(y2 - y1))
            if w >= 4 and h >= 4:
                new_idx = self.state.add_box_abs(
                    int(round(x1)),
                    int(round(y1)),
                    w,
                    h,
                    self.image_width,
                    self.image_height,
                )
                self.state.selected_box_idx = new_idx
                if self.on_boxes_changed:
                    self.on_boxes_changed()

        self.temp_draw_box = None
        self.drag_mode = None
        self.active_handle = None
        self.orig_box_on_drag = None
        self.redraw()

    def _on_mouse_move(self, event: tk.Event) -> None:
        """Update cursor according to handle or box hover."""
        if self.space_pressed or self.drag_mode == "pan":
            self.canvas.config(cursor="fleur")
            return

        cx, cy = event.x, event.y
        ix, iy = self.canvas_to_img(cx, cy)

        handle = self._find_handle_at(cx, cy)
        if handle:
            cursors = {
                "nw": "size_nw_se",
                "se": "size_nw_se",
                "ne": "size_ne_sw",
                "sw": "size_ne_sw",
                "n": "size_ns",
                "s": "size_ns",
                "e": "size_we",
                "w": "size_we",
            }
            self.canvas.config(cursor=cursors.get(handle, "arrow"))
            return

        box_idx = self._find_box_at(ix, iy)
        if box_idx is not None:
            self.canvas.config(cursor="fleur")
            return

        self.canvas.config(cursor="cross")

    def _on_delete_key(self, event: tk.Event) -> None:
        """Delete currently selected box (in-memory only per PLAN §7)."""
        if self.state.selected_box_idx is not None:
            self.state.remove_box(self.state.selected_box_idx)
            self.redraw()
            if self.on_boxes_changed:
                self.on_boxes_changed()

    # ---------------- Spatial Queries ----------------

    def _find_handle_at(self, cx: float, cy: float) -> str | None:
        """Return the handle name if cursor is over a handle of the selected box."""
        if self.state.selected_box_idx is None or self.original_image is None:
            return None

        abs_boxes = self.state.get_abs_boxes(self.image_width, self.image_height)
        if not (0 <= self.state.selected_box_idx < len(abs_boxes)):
            return None

        box = abs_boxes[self.state.selected_box_idx]
        x1, y1 = self.img_to_canvas(box["x"], box["y"])
        x2, y2 = self.img_to_canvas(box["x"] + box["w"], box["y"] + box["h"])

        positions = {
            "nw": (x1, y1),
            "n": ((x1 + x2) / 2, y1),
            "ne": (x2, y1),
            "e": (x2, (y1 + y2) / 2),
            "se": (x2, y2),
            "s": ((x1 + x2) / 2, y2),
            "sw": (x1, y2),
            "w": (x1, (y1 + y2) / 2),
        }

        h_thresh = HANDLE_SIZE
        for name, (hx, hy) in positions.items():
            if abs(cx - hx) <= h_thresh and abs(cy - hy) <= h_thresh:
                return name
        return None

    def _find_box_at(self, ix: float, iy: float) -> int | None:
        """Return index of box containing image coordinate (ix, iy)."""
        if self.original_image is None:
            return None

        abs_boxes = self.state.get_abs_boxes(self.image_width, self.image_height)
        # Search in reverse order to pick the topmost box
        for idx in range(len(abs_boxes) - 1, -1, -1):
            box = abs_boxes[idx]
            bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
            if bx <= ix <= bx + bw and by <= iy <= by + bh:
                return idx
        return None

    # ---------------- Debounced Preview (150ms per PLAN §7) ----------------

    def request_preview(self) -> None:
        """Schedule preview computation with 150ms debounce."""
        if self._preview_timer_id is not None:
            self.after_cancel(self._preview_timer_id)
        self._preview_timer_id = self.after(150, self._compute_and_show_preview)

    def _compute_and_show_preview(self) -> None:
        """Compute preview of logo removal on displayed image using current method and params."""
        self._preview_timer_id = None
        if self.original_image is None or self.image_width <= 0:
            return

        abs_boxes = self.state.get_abs_boxes(self.image_width, self.image_height)
        if not abs_boxes:
            self.preview_image = None
            self.show_preview = False
            self.redraw()
            return

        try:
            processed = apply(
                self.original_image,
                abs_boxes,
                self.state.method,
                self.state.params,
            )
            self.preview_image = processed
            self.show_preview = True
            self.redraw()
        except Exception as e:
            logger.warning("Preview generation failed: %s", e)
            self.show_preview = False
            self.redraw()

    def toggle_preview(self) -> bool:
        """Toggle between preview and original image. Returns current show_preview state."""
        if not self.show_preview:
            self._compute_and_show_preview()
            self.show_preview = True
        else:
            self.show_preview = False
            self.redraw()
        return self.show_preview
